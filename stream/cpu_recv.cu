/*
 * cpu_recv.cu -- kernel-stack (CPU-mediated) receiver for the streaming mode
 * (PU2).  The baseline of the streaming section: the same raw Ethertype
 * frames arrive through AF_PACKET sockets (the request service's socket
 * type, one per pinned CPU thread, PACKET_FANOUT round robin) into pinned
 * host memory, are copied to the GPU and consumed by the same accumulate
 * code as the GPUNetIO receiver.  PASS_END handling, update and PASS_ACK as
 * in the GPU receiver (the ACK goes out through the socket).
 *
 *   sudo ./bin/x86_64/st_cpu_recv -I ens7np0 -N 1000000 -B 8 -K 2 -t 8 -C 2 [-G]
 *
 * -G (count only): receive and discard, no GPU; the ceiling of the socket path.
 * Run it only while st_recv is NOT running (DOCA Flow would steal the Ethertype).
 */
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "stream_kernels.cuh"
#include "st_net.h"

using namespace st;

extern "C" int st_state_alloc(uint32_t n, uint32_t B, void **out, uint32_t **signs, int32_t **acc, long long **e2,
			      unsigned long long **chk);
extern "C" int st_state_init(void *p, uint32_t seed, cudaStream_t s);
extern "C" int st_state_update(void *p, uint32_t k, uint32_t K, float xi, float dt, int do_update, cudaStream_t s);

#define CR_BATCH 64u
#define CR_SLOTS 1024u   /* pinned ring slots per thread */

struct frame_meta { uint32_t row, cblk, kind; };

/* a batch of frames in device memory, processed by one block (the GPUNetIO
 * worker's logic: tile per column block, one warp per frame) */
template <int MAXB>
__global__ void __launch_bounds__(512) batch_kernel(const uint8_t *frames, const frame_meta *meta, uint32_t count,
						    const uint32_t *signs, int32_t *acc, uint32_t n, uint32_t B,
						    uint32_t wtot)
{
	extern __shared__ uint32_t tile[];
	__shared__ uint32_t seg_end;
	const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, nwarps = blockDim.x >> 5;
	uint32_t i = 0;
	while (i < count) {
		if (tid == 0) {
			uint32_t j = i + 1;
			while (j < count && meta[j].cblk == meta[i].cblk) ++j;
			seg_end = j;
		}
		__syncthreads();
		const uint32_t j = seg_end, cblk = meta[i].cblk;
		load_tile(tile, signs, wtot, B, cblk, tid, blockDim.x);
		__syncthreads();
		for (uint32_t f = i + warp; f < j; f += nwarps)
			if (meta[f].row < n)
				accumulate_frame<MAXB>((const uint32_t *)(frames + (size_t)f * ST_SLOT + ST_HDR_BYTES), tile, acc, n,
						       B, meta[f].row, lane);
		__syncthreads();
		i = j;
	}
}

struct shared_counts {
	std::atomic<unsigned long long> frames{0}, ends{0}, end_frames{0}, bad{0}, t_wait_ns{0};
	std::atomic<unsigned long long> gen_queues{0};
};

struct tctx {
	uint32_t id, n, B, wtot;
	int fd, core;
	bool count_only;
	uint8_t *pin;              /* pinned [CR_SLOTS][ST_SLOT] */
	uint8_t *dev;              /* device [2][CR_BATCH][ST_SLOT] */
	frame_meta *meta_pin, *meta_dev;   /* [2][CR_BATCH] */
	cudaStream_t s;
	uint32_t *signs; int32_t *acc;
	shared_counts *sc;
	std::atomic<bool> stop{false};
};

static int open_raw(const char *iface, int fanout_id, int *ifindex_out, uint8_t mac[6])
{
	int fd = socket(AF_PACKET, SOCK_RAW, htons(ST_ETHERTYPE));
	if (fd < 0) { perror("socket(AF_PACKET)"); return -1; }
	struct ifreq ifr; memset(&ifr, 0, sizeof(ifr));
	snprintf(ifr.ifr_name, IFNAMSIZ, "%s", iface);
	if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) { perror("SIOCGIFINDEX"); return -1; }
	*ifindex_out = ifr.ifr_ifindex;
	if (ioctl(fd, SIOCGIFHWADDR, &ifr) == 0) memcpy(mac, ifr.ifr_hwaddr.sa_data, 6);
	struct sockaddr_ll sll; memset(&sll, 0, sizeof(sll));
	sll.sll_family = AF_PACKET; sll.sll_protocol = htons(ST_ETHERTYPE); sll.sll_ifindex = *ifindex_out;
	if (bind(fd, (struct sockaddr *)&sll, sizeof(sll)) < 0) { perror("bind"); return -1; }
	int big = 256 << 20;
	setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));
	struct timeval tv = {0, 100000};
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#ifdef SO_BUSY_POLL
	int us = 50;
	setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &us, sizeof(us));
#endif
	if (fanout_id >= 0) {
		int fo = (fanout_id & 0xffff) | (PACKET_FANOUT_LB << 16);
		if (setsockopt(fd, SOL_PACKET, PACKET_FANOUT, &fo, sizeof(fo)) < 0) perror("PACKET_FANOUT");
	}
	return fd;
}

static void rx_thread(tctx *c)
{
	if (c->core >= 0) {
		cpu_set_t set; CPU_ZERO(&set); CPU_SET(c->core, &set);
		pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	}
	std::vector<struct mmsghdr> msgs(CR_BATCH);
	std::vector<struct iovec> iov(CR_BATCH);
	uint32_t buf = 0;
	unsigned long long head = 0;
	for (;;) {
		if (c->stop.load()) break;
		const uint32_t base = (uint32_t)(head % CR_SLOTS);
		const uint32_t room = CR_SLOTS - base < CR_BATCH ? CR_SLOTS - base : CR_BATCH;
		for (uint32_t i = 0; i < room; ++i) {
			iov[i].iov_base = c->pin + (size_t)(base + i) * ST_SLOT;
			iov[i].iov_len = ST_SLOT;
			memset(&msgs[i].msg_hdr, 0, sizeof(msgs[i].msg_hdr));
			msgs[i].msg_hdr.msg_iov = &iov[i];
			msgs[i].msg_hdr.msg_iovlen = 1;
		}
		const double t0 = st_now();
		const int got = recvmmsg(c->fd, msgs.data(), room, MSG_WAITFORONE, NULL);
		c->sc->t_wait_ns += (unsigned long long)((st_now() - t0) * 1e9);
		if (got <= 0) continue;
		uint32_t cnt = 0;
		frame_meta *meta = c->meta_pin + (size_t)buf * CR_BATCH;
		for (int i = 0; i < got; ++i) {
			const struct st_hdr *h = (const struct st_hdr *)(c->pin + (size_t)(base + i) * ST_SLOT);
			const uint32_t len = msgs[i].msg_len;
			meta[i].row = 0xFFFFFFFFu; meta[i].cblk = 0; meta[i].kind = 0;
			if (len < ST_HDR_BYTES || h->magic != ST_MAGIC) { c->sc->bad++; continue; }
			if (h->type == ST_PASS_END) {
				c->sc->gen_queues = h->gen_queues;
				c->sc->end_frames += h->seq;
				c->sc->ends += 1;
				continue;
			}
			if (h->type != ST_STREAM || len < ST_FRAME) { c->sc->bad++; continue; }
			meta[i].row = h->row; meta[i].cblk = h->cblk; meta[i].kind = 1;
			c->sc->frames += 1;
			cnt = (uint32_t)i + 1u;
		}
		if (cnt && !c->count_only) {
			cudaMemcpyAsync(c->dev + (size_t)buf * CR_BATCH * ST_SLOT, c->pin + (size_t)base * ST_SLOT,
					(size_t)cnt * ST_SLOT, cudaMemcpyHostToDevice, c->s);
			cudaMemcpyAsync(c->meta_dev + (size_t)buf * CR_BATCH, meta, sizeof(frame_meta) * CR_BATCH,
					cudaMemcpyHostToDevice, c->s);
			batch_kernel<ST_MAX_B><<<1, 512, c->B * ST_PAYLOAD, c->s>>>(c->dev + (size_t)buf * CR_BATCH * ST_SLOT,
				c->meta_dev + (size_t)buf * CR_BATCH, cnt, c->signs, c->acc, c->n, c->B, c->wtot);
			buf ^= 1u;
			/* pinned slots are reused after 16 batches; make sure their copies are done */
			if ((head / CR_BATCH) % 8 == 7) cudaStreamSynchronize(c->s);
		}
		head += got;
	}
}

int main(int argc, char **argv)
{
	uint32_t n = 10000, B = 8, K = 2, nt = 8, seed = 42, instance = ST_INST_MATTIS;
	int core_base = 2, dev = 0;
	float xi = 0.f, dt = ST_DEFAULT_DT;
	bool count_only = false;
	const char *iface = "ens7np0", *csv = NULL;
	uint8_t gen_mac[6] = {0x08, 0x20, 0xe7, 0x47, 0x10, 0x8e};
	for (int i = 1; i < argc; ++i) {
		std::string s = argv[i];
		auto next = [&](void) -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
		if (s == "-N") n = strtoul(next(), NULL, 10);
		else if (s == "-B") B = strtoul(next(), NULL, 10);
		else if (s == "-K") K = strtoul(next(), NULL, 10);
		else if (s == "-t") nt = strtoul(next(), NULL, 10);
		else if (s == "-s") seed = strtoul(next(), NULL, 10);
		else if (s == "-i") instance = std::string(next()) == "random" ? ST_INST_RANDOM : ST_INST_MATTIS;
		else if (s == "-x") xi = strtof(next(), NULL);
		else if (s == "-d") dt = strtof(next(), NULL);
		else if (s == "-I") iface = next();
		else if (s == "-m") st_parse_mac(next(), gen_mac);
		else if (s == "-C") core_base = atoi(next());
		else if (s == "-D") dev = atoi(next());
		else if (s == "-G") count_only = true;
		else if (s == "-o") csv = next();
		else { fprintf(stderr, "st_cpu_recv -I iface -N n -B B -K steps -t threads [-m gen MAC] [-C first core] [-G count-only] [-o csv]\n"); return 1; }
	}
	if (xi <= 0.f) xi = st_default_xi(n);
	cudaSetDevice(dev);
	void *S = NULL; uint32_t *signs = NULL; int32_t *acc = NULL; long long *e2 = NULL; unsigned long long *chk = NULL;
	if (!count_only) {
		if (st_state_alloc(n, B, &S, &signs, &acc, &e2, &chk)) return 1;
		st_state_init(S, seed, 0);
		cudaDeviceSynchronize();
		cudaFuncSetAttribute((const void *)batch_kernel<ST_MAX_B>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(B * ST_PAYLOAD));
		printf("# init chk 0x%016llx\n", *chk);
	}
	shared_counts sc;
	int ifindex = 0;
	uint8_t my_mac[6] = {0};
	const int fanout_id = (int)(getpid() & 0xffff);
	std::vector<tctx *> T(nt);
	std::vector<std::thread> th;
	for (uint32_t t = 0; t < nt; ++t) {
		tctx *c = new tctx;
		c->id = t; c->n = n; c->B = B; c->wtot = st_wtot(n); c->core = core_base >= 0 ? core_base + (int)t : -1;
		c->count_only = count_only; c->signs = signs; c->acc = acc; c->sc = &sc;
		c->fd = open_raw(iface, nt > 1 ? fanout_id : -1, &ifindex, my_mac);
		if (c->fd < 0) return 1;
		cudaHostAlloc((void **)&c->pin, (size_t)CR_SLOTS * ST_SLOT, cudaHostAllocDefault);
		cudaHostAlloc((void **)&c->meta_pin, sizeof(frame_meta) * 2 * CR_BATCH, cudaHostAllocDefault);
		memset(c->meta_pin, 0, sizeof(frame_meta) * 2 * CR_BATCH);
		if (!count_only) {
			cudaMalloc((void **)&c->dev, (size_t)2 * CR_BATCH * ST_SLOT);
			cudaMalloc((void **)&c->meta_dev, sizeof(frame_meta) * 2 * CR_BATCH);
			cudaStreamCreateWithFlags(&c->s, cudaStreamNonBlocking);
		}
		T[t] = c;
	}
	for (uint32_t t = 0; t < nt; ++t) th.emplace_back(rx_thread, T[t]);
	printf("# st_cpu_recv n %u B %u K %u threads %u %s; iface %s Ethertype 0x%04x; cores %d..%d\n", n, B, K, nt,
	       count_only ? "count-only" : "accumulate on GPU", iface, ST_ETHERTYPE, core_base, core_base + (int)nt - 1);
	FILE *fcsv = csv ? fopen(csv, "a") : NULL;
	if (fcsv) fprintf(fcsv, "pass,update,t_pass_s,payload_gbps,frames,expected,lost,bad,wait_frac,best_2E,chk\n");
	setvbuf(stdout, NULL, _IOLBF, 0);
	double t_pass0 = st_now(), t_rep = t_pass0;
	unsigned long long rep_frames = 0, prev_frames = 0, prev_expect = 0, prev_bad = 0, prev_wait = 0;
	int bad_run = 0;
	for (uint32_t pass = 0; pass <= K; ++pass) {
		if (!count_only) { *chk = 0ull; for (uint32_t b = 0; b < B; ++b) e2[b] = 0; }
		for (;;) {
			const unsigned long long gq = sc.gen_queues.load();
			if (gq && sc.ends.load() >= gq * (pass + 1ull)) break;
			const double now = st_now();
			if (now - t_rep >= 1.0) {
				const unsigned long long fr = sc.frames.load();
				printf("  t %8.1f s  pass %u  %.1f Gb/s payload  frames %llu  bad %llu\n", now - t_pass0, pass,
				       (double)(fr - rep_frames) * ST_PAYLOAD * 8.0 / (now - t_rep) / 1e9, fr, sc.bad.load());
				rep_frames = fr; t_rep = now;
			}
			usleep(200);
		}
		const double t_end = st_now();
		usleep(5000);   /* let the threads drain the tail of the pass */
		long long best = 0; unsigned long long ck = 0;
		if (!count_only) {
			for (auto *c : T) cudaStreamSynchronize(c->s);
			st_state_update(S, pass, K, xi, dt, pass < K, 0);
			cudaDeviceSynchronize();
			best = e2[0]; for (uint32_t b = 1; b < B; ++b) if (e2[b] < best) best = e2[b];
			ck = *chk;
		}
		const unsigned long long fr = sc.frames.load() - prev_frames, expect = sc.end_frames.load() - prev_expect;
		const unsigned long long bad = sc.bad.load() - prev_bad, wait = sc.t_wait_ns.load() - prev_wait;
		prev_frames = sc.frames.load(); prev_expect = sc.end_frames.load(); prev_bad = sc.bad.load(); prev_wait = sc.t_wait_ns.load();
		const long long lost = (long long)expect - (long long)fr;
		const double tp = t_end - t_pass0, gbps = (double)fr * ST_PAYLOAD * 8.0 / tp / 1e9;
		printf("pass %3u %s  t %9.3f s  %7.1f Gb/s  frames %llu/%llu lost %lld bad %llu  best 2E %lld  chk 0x%016llx  socket-wait %.1f%%\n",
		       pass, pass < K ? "upd" : "fin", tp, gbps, fr, expect, lost, bad, best, ck, 100.0 * (double)wait / (tp * 1e9 * nt));
		if (fcsv) fprintf(fcsv, "%u,%d,%.6f,%.3f,%llu,%llu,%lld,%llu,%.4f,%lld,0x%016llx\n", pass, pass < K, tp, gbps, fr, expect,
				  lost, bad, (double)wait / (tp * 1e9 * nt), best, ck);
		if (lost != 0 || bad) bad_run = 1;
		/* PASS_ACK through the socket (raw frame, as the client of the request service) */
		struct st_hdr ackh;
		st_build_hdr(&ackh, gen_mac, my_mac, ST_PASS_ACK, 0, pass, (uint32_t)(lost > 0 ? lost : 0), (uint32_t)fr,
			     (uint16_t)st_nblk(n), 0, (uint8_t)instance);
		struct sockaddr_ll sll; memset(&sll, 0, sizeof(sll));
		sll.sll_family = AF_PACKET; sll.sll_ifindex = ifindex; sll.sll_halen = 6; memcpy(sll.sll_addr, gen_mac, 6);
		sendto(T[0]->fd, &ackh, sizeof(ackh), 0, (struct sockaddr *)&sll, sizeof(sll));
		t_pass0 = st_now();
	}
	printf("# %s\n", bad_run ? "RUN INVALID: frames lost or bad" : "RUN OK: no frame lost");
	for (auto *c : T) c->stop = true;
	for (auto &t : th) t.join();
	if (fcsv) fclose(fcsv);
	return bad_run ? 2 : 0;
}

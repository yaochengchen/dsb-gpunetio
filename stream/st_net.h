/* st_net.h -- tiny host helpers shared by the generator, the receivers and
 * the kernel-stack receiver: MAC parsing, raw-frame header construction,
 * monotonic clock.  Plain C. */
#ifndef ST_NET_H_
#define ST_NET_H_

#include <arpa/inet.h>
#include <dirent.h>
#include <errno.h>
#include <net/if.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "stream_proto.h"

static inline int st_parse_mac(const char *s, uint8_t m[6])
{
	unsigned v[6];
	if (sscanf(s, "%x:%x:%x:%x:%x:%x", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6)
		return -1;
	for (int i = 0; i < 6; ++i)
		m[i] = (uint8_t)v[i];
	return 0;
}

/* Fills the 64-byte header of a raw frame. */
static inline void st_build_hdr(struct st_hdr *h, const uint8_t dmac[6], const uint8_t smac[6], uint8_t type,
				uint8_t queue, uint32_t pass, uint32_t row, uint32_t seq, uint16_t cblk, uint16_t gen_queues,
				uint8_t instance)
{
	memset(h, 0, sizeof(*h));
	memcpy(h->d_mac, dmac, 6);
	memcpy(h->s_mac, smac, 6);
	h->ethertype = htons(ST_ETHERTYPE);
	h->magic = ST_MAGIC;
	h->type = type;
	h->queue = queue;
	h->version = ST_VERSION;
	h->instance = instance;
	h->pass = pass;
	h->row = row;
	h->seq = seq;
	h->cblk = cblk;
	h->gen_queues = gen_queues;
}

static inline double st_now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC_RAW, &t);
	return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Kernel netdev name of the NIC at PCIe address `pci` (/sys/bus/pci/devices/<pci>/net/<if>); -1 if none. */
static inline int st_nic_ifname(const char *pci, char *ifname, size_t len)
{
	char path[160];
	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/net", pci);
	DIR *d = opendir(path);
	if (!d)
		return -1;
	struct dirent *e;
	int rc = -1;
	while ((e = readdir(d)) != NULL) {
		if (e->d_name[0] == '.')
			continue;
		snprintf(ifname, len, "%s", e->d_name);
		rc = 0;
		break;
	}
	closedir(d);
	return rc;
}

static inline int st_nic_get_mtu(const char *ifname)
{
	char path[160];
	int mtu = -1;
	snprintf(path, sizeof(path), "/sys/class/net/%s/mtu", ifname);
	FILE *f = fopen(path, "r");
	if (!f)
		return -1;
	if (fscanf(f, "%d", &mtu) != 1)
		mtu = -1;
	fclose(f);
	return mtu;
}

/* The frames of this protocol are ST_FRAME bytes: the port must be at MTU 9000
 * on BOTH ends (stream/README.md: ip link set <if> mtu 9000) or the NIC drops them as
 * oversize before any queue sees them.  Sets it when too small (needs root,
 * like the rest of the program); returns the MTU in effect or -1. */
static inline int st_nic_ensure_mtu(const char *pci, int need, char *ifname, size_t len)
{
	if (st_nic_ifname(pci, ifname, len))
		return -1;
	int mtu = st_nic_get_mtu(ifname);
	if (mtu >= need)
		return mtu;
	int s = socket(AF_INET, SOCK_DGRAM, 0);
	if (s >= 0) {
		struct ifreq ifr;
		memset(&ifr, 0, sizeof(ifr));
		snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", ifname);
		ifr.ifr_mtu = 9000;
		if (ioctl(s, SIOCSIFMTU, &ifr) == 0)
			fprintf(stderr, "[net] %s: MTU %d -> 9000 (frames are %d B)\n", ifname, mtu, need);
		else
			fprintf(stderr, "[net] %s: MTU %d too small for %d-byte frames and could not set 9000: %s\n", ifname, mtu,
				need, strerror(errno));
		close(s);
	}
	return st_nic_get_mtu(ifname);
}

#endif

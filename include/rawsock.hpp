// rawsock.hpp -- AF_PACKET raw Ethernet socket used by the client and by the
// CPU+GPU baseline server (both need CAP_NET_RAW, i.e. sudo).
#pragma once

#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "dsb_host.hpp"
#include "dsb_proto.h"

struct RawSock {
  int fd = -1;
  int ifindex = 0;
  dsbhost::Mac mac{};
  int mtu = 0;

  void open(const std::string& iface, bool busy_poll = true) {
    fd = ::socket(AF_PACKET, SOCK_RAW, htons(DSB_ETHERTYPE));
    if (fd < 0) throw std::runtime_error("socket(AF_PACKET): " + std::string(strerror(errno)) + " (run as root)");
    struct ifreq ifr;
    std::memset(&ifr, 0, sizeof(ifr));
    std::snprintf(ifr.ifr_name, IFNAMSIZ, "%s", iface.c_str());
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) throw std::runtime_error("SIOCGIFINDEX " + iface);
    ifindex = ifr.ifr_ifindex;
    if (ioctl(fd, SIOCGIFHWADDR, &ifr) < 0) throw std::runtime_error("SIOCGIFHWADDR " + iface);
    std::memcpy(mac.b, ifr.ifr_hwaddr.sa_data, 6);
    if (ioctl(fd, SIOCGIFMTU, &ifr) == 0) mtu = ifr.ifr_mtu;
    struct sockaddr_ll sll;
    std::memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_protocol = htons(DSB_ETHERTYPE);
    sll.sll_ifindex = ifindex;
    if (bind(fd, reinterpret_cast<sockaddr*>(&sll), sizeof(sll)) < 0)
      throw std::runtime_error("bind: " + std::string(strerror(errno)));
    int one = 1;
    setsockopt(fd, SOL_PACKET, PACKET_QDISC_BYPASS, &one, sizeof(one));  // skip the qdisc on TX
    int big = 64 << 20;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &big, sizeof(big));
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &big, sizeof(big));
#ifdef SO_BUSY_POLL
    if (busy_poll) {
      int us = 50;
      setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &us, sizeof(us));
    }
#endif
  }

  /// Non-blocking receive of one frame; returns bytes or 0.
  ssize_t recv(uint8_t* buf, size_t cap) {
    const ssize_t r = ::recv(fd, buf, cap, MSG_DONTWAIT);
    if (r < 0) return 0;
    return r;
  }

  int last_errno = 0;      // errno of the last failed send
  uint64_t send_fail = 0;  // frames that could not be sent

  /// Largest frame this interface will accept from AF_PACKET (MTU + 14-byte header).
  size_t max_frame() const { return size_t(mtu > 0 ? mtu : 1500) + 14; }

  bool send(const uint8_t* f, size_t len) {
    struct sockaddr_ll sll;
    std::memset(&sll, 0, sizeof(sll));
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = ifindex;
    sll.sll_halen = 6;
    std::memcpy(sll.sll_addr, f, 6);
    for (;;) {
      const ssize_t r = ::sendto(fd, f, len, 0, reinterpret_cast<sockaddr*>(&sll), sizeof(sll));
      if (r == ssize_t(len)) return true;
      if (r < 0 && (errno == ENOBUFS || errno == EAGAIN || errno == EINTR)) continue;
      last_errno = r < 0 ? errno : EIO;
      ++send_fail;
      return false;
    }
  }

  const char* send_error() const { return std::strerror(last_errno); }

  ~RawSock() {
    if (fd >= 0) ::close(fd);
  }
};

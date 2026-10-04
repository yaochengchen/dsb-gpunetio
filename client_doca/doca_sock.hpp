#pragma once
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include "cdoca.h"
#include "dsb_host.hpp"
struct DocaSock {
  dsbhost::Mac mac{};
  int mtu = 9000;
  bool up = false;
  void open(const std::string& gpu, const std::string& nic) {
    if (cdoca_open(gpu.c_str(), nic.c_str(), mac.b) != 0) throw std::runtime_error("cdoca_open failed (see [cdoca] lines)");
    up = true;
  }
  ssize_t recv(uint8_t* b, size_t cap) { return cdoca_recv(b, uint32_t(cap)); }
  bool send(const uint8_t* f, size_t len) { return cdoca_send(f, uint32_t(len)) == 1; }
  ~DocaSock() { if (up) cdoca_close(); }
};

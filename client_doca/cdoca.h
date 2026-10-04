#ifndef CDOCA_H_
#define CDOCA_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int  cdoca_open(const char *gpu_pci, const char *nic_pci, uint8_t mac[6]);
int  cdoca_send(const uint8_t *frame, uint32_t len);   /* 1 = queued */
int  cdoca_recv(uint8_t *buf, uint32_t cap);           /* bytes, 0 = none */
void cdoca_close(void);
#ifdef __cplusplus
}
#endif
#endif

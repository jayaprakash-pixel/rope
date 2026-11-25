/* Hardware probe: what this device can do. The planner decides everything from these numbers. */
#pragma once
#include <stdint.h>

#define ROPE_MAX_GPUS 8

typedef struct {
    char     name[128];
    int      vendor;            /* PCI vendor id: 0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel */
    uint64_t dedicated_bytes;   /* VRAM (0 for most integrated GPUs) */
    uint64_t shared_bytes;      /* system RAM the GPU may borrow */
    int      integrated;
} rope_adapter;

typedef struct {
    /* system */
    char     os[64];
    char     os_name[96];       /* e.g. "Windows 11 Home Single Language 24H2" (registry, read only) */
    char     machine[96];       /* maker and model from the firmware strings, e.g. "ASUSTeK COMPUTER INC. TUF Gaming A15" (no serial number) */
    int      cpu_mhz;           /* nominal clock (registry), 0 = unknown */
    int      on_ac;             /* 1 mains power, 0 battery, -1 unknown / no battery */
    int      battery_pct;       /* -1 unknown */
    int      battery_saver;     /* 1 Windows battery saver is on, 0 off, -1 unknown */
    uint64_t ram_installed;    /* physically installed (firmware view) */
    uint64_t ram_total, ram_free; /* usable by Windows (installed minus firmware / iGPU reservations) */
    /* cpu */
    char     cpu_name[64];
    int      cores, threads;
    int      sse42, avx, avx2, fma, f16c, avx512f, avx512_vnni, avx_vnni;
    /* NVIDIA CUDA device 0 (via the driver); cuda_ok == 0 means none usable */
    int      cuda_ok, cuda_driver, cc_major, cc_minor, sm_count, clock_khz;
    double   h2d_gbs;                 /* pinned host -> VRAM copy speed, GB/s (0 = not measured) */
    char     cuda_name[128];
    uint64_t vram_total, vram_free;
    /* all display adapters (DXGI) */
    int      n_adapters;
    rope_adapter adapters[ROPE_MAX_GPUS];
    /* storage of the probed directory */
    char     disk_dir[260];
    uint64_t disk_free, disk_total;
    int      disk_ssd;          /* 1 SSD, 0 HDD, -1 unknown */
    char     disk_bus[16];      /* "NVMe", "SATA", "USB", ... */
    double   disk_read_gbs;     /* sequential, OS cache bypassed; 0 if not measured */
    double   disk_write_gbs;
} rope_probe;

/* dir: directory whose disk is measured (UTF-8). disk_test_mb: size of the temporary test file, 0 = skip speed test. */
int  rope_probe_run(rope_probe *p, const char *dir, int disk_test_mb);
void rope_probe_print_json(const rope_probe *p);

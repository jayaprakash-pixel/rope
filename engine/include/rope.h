/* librope: run image models on low-end laptops, from C, C++, Python (ctypes), or any language that can call a C library.
 *
 * Link with librope.dll (Windows: librope.dll.lib). Every function is plain C. Requests and answers are JSON text.
 * Rules:
 *  - Strings you pass in are NUL-terminated UTF-8. Strings come out into your buffer (char *out, size_t cap) with the snprintf rule:
 *    the text is cut to fit and always ends with a NUL; the return value is the FULL length (so a result >= cap means "too small"),
 *    or -1 on error (read the reason with rope_last_error).
 *  - One image job runs at a time in a process. rope_job_start returns NULL while another job runs.
 *  - The progress callback runs on the job's own thread. Do not call rope_job_free from inside it.
 *  - Downloads are off unless the request has "allow_download": true. A missing model file is then an error, not a download.
 *  - The library writes plan and timing lines to stderr (the same as rope.exe).
 */
#ifndef ROPE_H
#define ROPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  define ROPE_API __declspec(dllimport)
#else
#  define ROPE_API
#endif

typedef struct rope_job rope_job;

/* stage: "text", "denoise", "decode" or "upscale"; step i of n. */
typedef void (*rope_progress_fn)(const char *stage, int i, int n, void *user);

/* Library version, e.g. "0.1.0". The string is static. */
ROPE_API const char *rope_version(void);

/* Text of the last error of a failed call. */
ROPE_API int rope_last_error(char *out, size_t cap);

/* This laptop as JSON: CPU, RAM, GPU, disk (same as `rope probe`). dir: a folder on the disk to measure, or NULL. */
ROPE_API int rope_probe_json(const char *dir, char *out, size_t cap);

/* Every catalog model with its plan for this laptop, as JSON (same as `rope suggest --json`).
 * request (JSON, all fields optional): models_dir, catalog, quality ("fast"|"balanced"|"best"), size, only (a model id),
 * upscale (true: plan with the 4x upscaler), no_gpu, ram_gib (pretend a smaller laptop). */
ROPE_API int rope_plan_json(const char *request, char *out, size_t cap);

/* Outlier-injection scan of a model or add-on file. JSON: {"tensors", "groups", "flagged": [names], "clean": true|false}. */
ROPE_API int rope_scan_json(const char *path, char *out, size_t cap);

/* Start one image in a thread. request (JSON):
 *   prompt          required
 *   model           path of a model file; without it rope picks the best catalog model for this laptop
 *   quality         "fast" | "balanced" (default) | "best"
 *   seed, size, steps
 *   out             path of the PNG to write (default "rope.png")
 *   models_dir, catalog   where the models and the catalog are (defaults "models", "catalog/models.json")
 *   lora, lora_scale      a LoRA add-on file and its strength (SD 1.x / 2.x / SDXL models)
 *   upscale               a Real-ESRGAN x4 add-on file: the image is 4 times larger on each side
 *   no_gpu, ram_gib, allow_download
 * Returns NULL if the request is wrong or a job runs. cb may be NULL. */
ROPE_API rope_job *rope_job_start(const char *request, rope_progress_fn cb, void *user);

/* Wait for the end of the job. Returns 0 (the PNG is written), 1 (failed: msg has the reason) or 2 (cancelled). msg may be NULL. */
ROPE_API int rope_job_wait(rope_job *job, char *msg, size_t cap);

/* Ask the job to stop at its next step (within one denoising step or one tile), then call rope_job_wait. */
ROPE_API void rope_job_cancel(rope_job *job);

/* Free the job. If it still runs, it is cancelled and waited for. */
ROPE_API void rope_job_free(rope_job *job);

#ifdef __cplusplus
}
#endif
#endif

/* One image through librope from C.
 *   librope_example MODEL_FILE OUT.png "prompt" [SEED]
 * Build (MSVC):  cl /I engine\include examples\librope_example.c build\rust-cpu\release\librope.dll.lib
 * Run with librope.dll next to the program or on PATH. */
#include <stdio.h>
#include <stdlib.h>
#include "rope.h"

static void progress(const char *stage, int i, int n, void *user) {
    (void)user;
    printf("  %s %d/%d\n", stage, i, n);
    fflush(stdout);
}

/* Paths go into JSON text: use forward slashes (Windows accepts them) so no backslash needs escaping. */
static char *slashes(char *p) { for (char *c = p; *c; c++) if (*c == '\\') *c = '/'; return p; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL_FILE OUT.png \"prompt\" [SEED]\n", argv[0]); return 2; }
    printf("librope %s\n", rope_version());

    static char plan[1 << 17];
    int n = rope_plan_json("{\"quality\": \"fast\", \"no_gpu\": true}", plan, sizeof plan);
    if (n < 0 || n >= (int)sizeof plan) { fprintf(stderr, "plan failed\n"); return 1; }
    printf("plan JSON: %d bytes\n", n);

    char req[2048];
    snprintf(req, sizeof req, "{\"model\": \"%s\", \"prompt\": \"%s\", \"out\": \"%s\", \"quality\": \"fast\", \"size\": 256, \"steps\": 1, \"seed\": %s, \"no_gpu\": true}",
             slashes(argv[1]), argv[3], slashes(argv[2]), argc > 4 ? argv[4] : "7");
    rope_job *job = rope_job_start(req, progress, NULL);
    if (!job) { char e[512]; rope_last_error(e, sizeof e); fprintf(stderr, "start failed: %s\n", e); return 1; }
    char msg[512];
    int rc = rope_job_wait(job, msg, sizeof msg);
    rope_job_free(job);
    if (rc != 0) { fprintf(stderr, "job %s: %s\n", rc == 2 ? "cancelled" : "failed", msg); return 1; }
    printf("wrote %s\n", argv[2]);
    return 0;
}

#include <inttypes.h>
#include <stdlib.h>

#include "ggml.h"

#define GGML_MAX_NAME_LEN 64

#define GGML_PROFILE_NONE    0  // 0000 0000
#define GGML_PROFILE_COMPUTE 1  // 0000 0001 (1 << 0)
#define GGML_PROFILE_MEMORY  2  // 0000 0010 (1 << 1)
#define GGML_PROFILE_DISK 4     // 0000 0100 (1 << 2)

/* extern "C" keyword is needed
   because the way mangling symbol is different
   for C and C++
 */
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GGML_PROFILE_NONE    = 0,
  GGML_PROFILE_GRAPH   = 1 << 0, // 1 
  GGML_PROFILE_COMPUTE = 1 << 1, // 2
} GGML_PROFILE_MODE;

typedef struct ggml_profile_node_record_t {
  int step;
  char node_name[GGML_MAX_NAME_LEN];
  double node_compute_time_ns;
  size_t node_tensor_size_bytes;
} ggml_profile_node_record_t;

typedef struct ggml_profile_manager_t {
  // Profiling Setup
  GGML_PROFILE_MODE profile_mode;
  char *output_dir;
  double tsc_period_ns;
  // Decode Step
  int step; 
  // Array of recorded data.
  size_t record_cap;
  size_t record_size;
  ggml_profile_node_record_t *record_arr;
  // Record the starting timestamp
  uint64_t tmp_tsc;
  struct timespec tmp_ts;
} ggml_profile_manager_t;

// Singleton object
extern ggml_profile_manager_t ggml_profile_manager;

void ggml_profile_init(void);
void ggml_profile_quit(void);
static bool ggml_profile_node(struct ggml_tensor *t, bool ask, void *user_data);

double ggml_profile_tsc_calibration(void);
static inline uint64_t ggml_profile_tsc_get(void);

#ifdef __cplusplus
}
#endif

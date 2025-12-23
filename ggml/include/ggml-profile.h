#include <inttypes.h>
#include <stdlib.h>

#include "ggml.h"

#define GGML_MAX_NAME_LEN 64

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
  char node_src_name[GGML_MAX_NAME_LEN];
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
GGML_API ggml_profile_manager_t ggml_profile_manager;

GGML_API void ggml_profile_init(void);
GGML_API void ggml_profile_quit(void);
GGML_API bool ggml_profile_node(struct ggml_tensor *t, bool ask, void *user_data);

#ifdef __cplusplus
}
#endif

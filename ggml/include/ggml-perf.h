#include <inttypes.h>
#include <stdlib.h>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>


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
  GGML_PROFILE_PAGE_FAULT   = 1 << 0, // 1 
} GGML_PROFILE_MODE;

typedef struct ggml_profile_pf_record_t {
  int step;
  long long major_page_faults;
  long long minor_page_faults;
} ggml_profile_pf_record_t;

typedef struct ggml_profile_manager_t {
  // Profiling Setup
  GGML_PROFILE_MODE profile_mode;
  char *output_dir;
  // Decode Step
  int step;
  // Array of recorded data.
  size_t record_cap;
  size_t record_size;
  ggml_profile_pf_record_t *record_arr;
  // File descriptor for perf interface
  int fd;
} ggml_profile_manager_t;

// Singleton object
GGML_API ggml_profile_manager_t ggml_profile_manager;

GGML_API void ggml_profile_init(void);
GGML_API void ggml_profile_quit(void);
GGML_API bool ggml_profile_pre_token(void);
GGML_API bool ggml_profile_post_token(void);

#ifdef __cplusplus
}
#endif

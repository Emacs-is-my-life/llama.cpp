#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-profile.h"

// Singleton object
ggml_profile_manager_t ggml_profile_manager;

// Compute Profiling Code
#if defined(__x86_64__)

#include <x86intrin.h>

double ggml_profile_tsc_calibration(void) {
  struct timespec ts_wait_period = { .tv_sec  = 0,
                                     .tv_nsec = 100000000}; // Sleep for 100ms

  struct timespec ts_start;
  struct timespec ts_end;
  uint64_t tsc_start;
  uint64_t tsc_end;

  bool isCalibrated = false;

  while (!isCalibrated) {
	unsigned int aux_start;
    unsigned int aux_end;
          
    // Get Walltime
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    tsc_start = __rdtscp(&aux_start);

    // Forward Barrier
    __asm__ __volatile__("lfence" ::: "memory");

    // Sleep
    nanosleep(&ts_wait_period, NULL);

    // Getll Walltime & TSC
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
    tsc_end = __rdtscp(&aux_end);

    // Forward Barrier
    __asm__ __volatile__("lfence" ::: "memory");

    if (aux_start == aux_end) {
	  isCalibrated = true;
    }
  }

  uint64_t ns_elapsed = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000L +
                        (ts_end.tv_nsec - ts_start.tv_nsec);
  
  uint64_t tsc_elapsed = tsc_end - tsc_start;

  double tsc_period_in_ns = (double)ns_elapsed / (double)tsc_elapsed;
  return tsc_period_in_ns;
}

static inline uint64_t ggml_profile_tsc_get(void) {
  unsigned int aux;
  uint64_t tsc;

  tsc = __rdtscp(&aux);
  __asm__ __volatile__("lfence" ::: "memory");

  return tsc;
}

#elif defined(__aarch64__)

double ggml_profile_tsc_calibration(void) {
  struct timespec ts_wait_period = { .tv_sec  = 0,
                                     .tv_nsec = 100000000}; // Sleep for 100ms

  struct timespec ts_start;
  struct timespec ts_end;
  uint64_t tsc_start;
  uint64_t tsc_end;
          
  // Get Walltime 
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
  // Get TSC
  __asm__ __volatile__ ("isb" ::: "memory");
  __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r" (tsc_start));
  __asm__ __volatile__ ("isb" ::: "memory");

  // Sleep
  nanosleep(&ts_wait_period, NULL);

  // Get Walltime
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);
  // Get TSC
  __asm__ __volatile__ ("isb" ::: "memory");
  __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r" (tsc_end));
  __asm__ __volatile__ ("isb" ::: "memory");

  uint64_t ns_elapsed = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000L +
                        (ts_end.tv_nsec - ts_start.tv_nsec);
  
  uint64_t tsc_elapsed = tsc_end - tsc_start;

  double tsc_period_in_ns = (double)ns_elapsed / (double)tsc_elapsed;
  return tsc_period_in_ns;
}

static inline uint64_t ggml_profile_tsc_get(void) {
  uint64_t tsc;

  __asm__ __volatile__ ("isb" ::: "memory");
  __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r" (tsc));
  __asm__ __volatile__ ("isb" ::: "memory");

  return tsc;
}

#endif

void ggml_profile_init(void) {
  // Set profiling mode
  GGML_PROFILE_MODE profile_mode = GGML_PROFILE_NONE;

  if (getenv("GGML_PROFILE_GRAPH") != NULL) {
	profile_mode |= GGML_PROFILE_GRAPH;
  }

  if (getenv("GGML_PROFILE_COMPUTE") != NULL) {
    profile_mode |= GGML_PROFILE_COMPUTE;
  }

  ggml_profile_manager.profile_mode = profile_mode;

  // Set output directory
  char* ggml_profile_output_dir = getenv("GGML_PROFILE_OUTPUT_DIR");
  if (ggml_profile_output_dir == NULL) {
	ggml_profile_output_dir = "./profile-output";
  }
  ggml_profile_manager.output_dir = ggml_profile_output_dir;

  if (ggml_profile_manager.profile_mode != 0) {
	struct stat sb;
	if (stat(ggml_profile_output_dir, &sb)) {
	  if (mkdir(ggml_profile_output_dir, 0755) != 0) {
		perror("Failed to create ggml profile output directory.");
		exit(EXIT_FAILURE);
	  }    
	}
  }
  
  // TSC Calibrarion (for AMD64 or AARCH64)
#if defined(__x86_64__) || defined(__aarch64__)
  ggml_profile_manager.tsc_period_ns = ggml_profile_tsc_calibration();
#else
  ggml_profile_manager.tsc_period_ns = -1;
#endif

  // Set step = 0
  ggml_profile_manager.step = 0;

  // Initialize array to hold ggml_profile_node_records
  ggml_profile_manager.record_cap  = 4096;  // Set initial capacity to 4096
  ggml_profile_manager.record_size = 0;    // Empty
  ggml_profile_manager.record_arr = malloc(ggml_profile_manager.record_cap *
                                           sizeof(ggml_profile_node_record_t));
  if (ggml_profile_manager.record_arr == NULL) {
    perror("Failed to allocate memory for ggml profile records.");
	exit(EXIT_FAILURE);
  }
}

void ggml_profile_record_append(const ggml_profile_node_record_t *node_record) {
  // Check if record_arr is full.
  // If it's full, then realloc to stretch it.
  bool is_record_arr_full = ggml_profile_manager.record_size >= ggml_profile_manager.record_cap;
  if (is_record_arr_full) {
    size_t new_record_cap = 2 * ggml_profile_manager.record_cap;
    ggml_profile_node_record_t *new_record_arr =
        realloc(ggml_profile_manager.record_arr,
                new_record_cap * sizeof(ggml_profile_node_record_t));
    if (new_record_arr == NULL) {
      perror("Failed to re-allocate memory for ggml");
	  exit(EXIT_FAILURE);
    }

    ggml_profile_manager.record_arr = new_record_arr;
	ggml_profile_manager.record_cap = new_record_cap;
  }

  // Append the new record
  ggml_profile_manager.record_arr[ggml_profile_manager.record_size] =
      *node_record;
  ggml_profile_manager.record_size++;
}

void ggml_profile_record_write(void) {
  // Set write path
  char output_path[512];
  strcpy(output_path, ggml_profile_manager.output_dir);
  strcat(output_path, "/ggml_profile_node_records.csv");

  FILE *f_ptr = fopen(output_path, "w");
  if (f_ptr == NULL) {
    perror("Failed to open output file for profile records.");
	exit(EXIT_FAILURE);
  }

  // Write node_records in CSV format
  fprintf(f_ptr, "step,node_n,node_name,tensor_addr,node_compute_time_ns,node_tensor_size_bytes,node_input_size_bytes\n");
  size_t record_size = ggml_profile_manager.record_size;
  ggml_profile_node_record_t* record_arr = ggml_profile_manager.record_arr;
  for (size_t i = 0; i < record_size; i++) {
	ggml_profile_node_record_t* node_record = &record_arr[i];
	fprintf(f_ptr, "%d,%d,%s,0x%" PRIxPTR ",%lf,%zu,%zu\n",
                node_record->step,
                node_record->node_n,
                node_record->node_name,
                node_record->tensor_addr,
                node_record->node_compute_time_ns,
                node_record->node_tensor_size_bytes,
				node_record->node_input_size_bytes);
  }

  fflush(f_ptr);
  fclose(f_ptr);
}

void ggml_profile_quit(void) {
  // Write node profiling records to disk
  GGML_PROFILE_MODE ggml_profile_mode = ggml_profile_manager.profile_mode;
  bool records_exist = (ggml_profile_mode & GGML_PROFILE_GRAPH) ||
                       (ggml_profile_mode & GGML_PROFILE_COMPUTE);
  if (records_exist) {
	ggml_profile_record_write();
  }

  // Free memory
  free(ggml_profile_manager.record_arr);
}

struct ggml_tensor * ggml_profile_get_concrete_tensor_addr(struct ggml_tensor *t) {
  if (t->view_src == NULL) {
	return t;
  }
  
  return ggml_profile_get_concrete_tensor_addr(t->view_src);
}

bool ggml_profile_node(struct ggml_tensor *t, bool ask,
                              void *user_data) {
  GGML_PROFILE_MODE ggml_profile_mode = ggml_profile_manager.profile_mode;
  if (ggml_profile_mode == GGML_PROFILE_NONE) {
	return true;
  }
  
  if (ask) { // Pre-execution hook
    if (ggml_profile_mode & GGML_PROFILE_COMPUTE) { // Compute Profile
                                                    
#if defined(__x86_64__) || defined(__aarch64__)     // TSC available
      uint64_t tsc_start = ggml_profile_tsc_get();
	  ggml_profile_manager.tmp_tsc = tsc_start;
#else                                               // TSC not available
      struct timespec ts_start;
      clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
	  ggml_profiler_manager.tmp_ts = ts_start;
#endif
	  
    }
  } else {  // Post-execution hook
    if (ggml_profile_mode & GGML_PROFILE_COMPUTE) { // Compute Profile
      double node_compute_time_ns = -1;
	  
#if defined(__x86_64__) || defined(__aarch64__)     // TSC available
      uint64_t tsc_end = ggml_profile_tsc_get();
      uint64_t tsc_start = ggml_profile_manager.tmp_tsc;

      node_compute_time_ns = (tsc_end - tsc_start) * ggml_profile_manager.tsc_period_ns;
#else                                               // TSC not available
      struct timespec ts_end;
      clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

      node_compute_time_ns = (ts_end.tv_sec - ts_start.tv_sec) * 1000000000L +
                             (ts_end.tv_nsec - ts_start.tv_nsec);
#endif

      ggml_profile_node_record_t node_record;
      node_record.step = ggml_profile_manager.step;
	  node_record.node_n = ggml_profile_manager.tmp_node_n++;
	  strcpy(node_record.node_name, t->name);
	  node_record.tensor_addr = (uintptr_t) ggml_profile_get_concrete_tensor_addr(t);
      node_record.node_compute_time_ns = node_compute_time_ns;
      node_record.node_tensor_size_bytes = ggml_nbytes_pad(t);
      size_t node_input_size_bytes = 0;
      for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor *src = t->src[i];
        if (src == NULL) {
		  break;
        }

		node_input_size_bytes += ggml_nbytes_pad(src);
      }
	  node_record.node_input_size_bytes = node_input_size_bytes;

      ggml_profile_record_append(&node_record);
	}
  }

  return true;
}

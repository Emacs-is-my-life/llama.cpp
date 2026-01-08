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
#include "ggml-perf.h"

// Singleton object
ggml_profile_manager_t ggml_profile_manager;

static long perf_event_open(struct perf_event_attr *hw_event, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, 0, -1, -1, flags);
}

int setup_perf_counter(uint64_t config) {
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(struct perf_event_attr));

  pe.type = PERF_TYPE_SOFTWARE;
  pe.size = sizeof(struct perf_event_attr);
  pe.config = config;
  pe.disabled = 1;        // Start disabled
  pe.exclude_kernel = 1;  // Exclude kernel-space faults
  pe.exclude_hv = 1;      // Exclude hypervisor faults

  int fd = perf_event_open(&pe, 0);
  if (fd == -1) {
	perror("Failed to create perf event counter");
	exit(EXIT_FAILURE);
  }

  return fd;
}


void ggml_profile_init(void) {
  // Set profiling mode
  GGML_PROFILE_MODE profile_mode = GGML_PROFILE_NONE;

  if (getenv("GGML_PROFILE_PAGE_FAULT") != NULL) {
	profile_mode |= GGML_PROFILE_PAGE_FAULT;
  }

  ggml_profile_manager.profile_mode = profile_mode;
  ggml_profile_manager.step = 0;

  // Set output directory
  char* ggml_profile_output_dir = getenv("GGML_PROFILE_OUTPUT_DIR");
  if (ggml_profile_output_dir == NULL) {
	ggml_profile_output_dir = "./ggml_profile_output";
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

  // Initialize perf counters
  if (profile_mode & GGML_PROFILE_PAGE_FAULT) {
	ggml_profile_manager.fd_perf_major_page_faults = setup_perf_counter(PERF_COUNT_SW_PAGE_FAULTS_MAJ);
	ggml_profile_manager.fd_perf_major_page_faults = setup_perf_counter(PERF_COUNT_SW_PAGE_FAULTS_MIN);
  }
}

void ggml_profile_record_append(const ggml_profile_pf_record_t *pf_record) {
  // Check if record_arr is full.
  // If it's full, then realloc to stretch it.
  bool is_record_arr_full = ggml_profile_manager.record_size >= ggml_profile_manager.record_cap;
  if (is_record_arr_full) {
    size_t new_record_cap = 2 * ggml_profile_manager.record_cap;
    ggml_profile_pf_record_t *new_record_arr =
        realloc(ggml_profile_manager.record_arr,
                new_record_cap * sizeof(ggml_profile_pf_record_t));
    if (new_record_arr == NULL) {
      perror("Failed to re-allocate memory for ggml");
	  exit(EXIT_FAILURE);
    }

    ggml_profile_manager.record_arr = new_record_arr;
	ggml_profile_manager.record_cap = new_record_cap;
  }

  // Append the new record
  ggml_profile_manager.record_arr[ggml_profile_manager.record_size] = *pf_record;
  ggml_profile_manager.record_size++;
}

void ggml_profile_record_write(void) {
  // Set write path
  char output_path[512];
  strcpy(output_path, ggml_profile_manager.output_dir);
  strcat(output_path, "/ggml_profile_page_faults.csv");

  FILE *f_ptr = fopen(output_path, "w");
  if (f_ptr == NULL) {
    perror("Failed to open output file for profile records.");
	exit(EXIT_FAILURE);
  }

  // Write node_records in CSV format
  fprintf(f_ptr, "step,major_page_faults,minor_page_faults\n");
  size_t record_size = ggml_profile_manager.record_size;
  ggml_profile_pf_record_t* record_arr = ggml_profile_manager.record_arr;
  for (size_t i = 0; i < record_size; i++) {
	ggml_profile_pf_record_t* pf_record = &record_arr[i];
	fprintf(f_ptr, "%d,%lld,%lld\n",
                pf_record->step,
                pf_record->major_page_faults,
                pf_record->minor_page_faults);
  }

  fflush(f_ptr);
  fclose(f_ptr);
}

void ggml_profile_quit(void) {
  // Write node profiling records to disk
  GGML_PROFILE_MODE ggml_profile_mode = ggml_profile_manager.profile_mode;
  bool records_exist = (ggml_profile_mode & GGML_PROFILE_PAGE_FAULT);
  if (records_exist) {
	ggml_profile_record_write();

	close(ggml_profile_manager.fd_perf_major_page_faults);
	close(ggml_profile_manager.fd_perf_minor_page_faults);
  }

  // Free memory
  free(ggml_profile_manager.record_arr);
}

void ggml_profile_pre_token(void) {
  // Reset counters
  ioctl(ggml_profile_manager.fd_perf_major_page_faults, PERF_EVENT_IOC_RESET, 0);
  ioctl(ggml_profile_manager.fd_perf_minor_page_faults, PERF_EVENT_IOC_RESET, 0);

  // Enable counters
  ioctl(ggml_profile_manager.fd_perf_major_page_faults, PERF_EVENT_IOC_ENABLE, 0);
  ioctl(ggml_profile_manager.fd_perf_minor_page_faults, PERF_EVENT_IOC_ENABLE, 0);
}

void ggml_profile_post_token(void) {
  // Disable counters
  ioctl(ggml_profile_manager.fd_perf_major_page_faults, PERF_EVENT_IOC_DISABLE, 0);
  ioctl(ggml_profile_manager.fd_perf_major_page_faults, PERF_EVENT_IOC_DISABLE, 0);

  long long pf_major = 0;
  long long pf_minor = 0;

  read(ggml_profile_manager.fd_perf_major_page_faults, &pf_major, sizeof(long long));
  read(ggml_profile_manager.fd_perf_minor_page_faults, &pf_minor, sizeof(long long));

  ggml_profile_pf_record_t pf_record;
  pf_record.step = ggml_profile_manager.step;
  pf_record.major_page_faults = pf_major;
  pf_record.minor_page_faults = pf_minor;
  ggml_profile_record_append(&pf_record);

  ggml_profile_manager.step++;
}

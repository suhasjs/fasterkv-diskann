set(DISKANN_HEADERS
  graph.h
)

add_executable(in_memory_search ${DISKANN_HEADERS} in_memory_search.cc)
target_link_libraries(in_memory_search ${FASTER_BENCHMARK_LINK_LIBS})


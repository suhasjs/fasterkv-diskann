set(DISKANN_HEADERS
  mem_graph.h disk_graph.h
)

add_executable(in_memory_search ${DISKANN_HEADERS} in_memory_search.cc)
add_executable(zero_memory_search ${DISKANN_HEADERS} zero_memory_search.cc)
target_link_libraries(in_memory_search ${FASTER_BENCHMARK_LINK_LIBS})
target_link_libraries(zero_memory_search ${FASTER_BENCHMARK_LINK_LIBS})


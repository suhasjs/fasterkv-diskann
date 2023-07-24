set(DISKANN_HEADERS
  mem_graph.h disk_graph.h mutable_mem_graph.h
)

add_executable(in_memory_search ${DISKANN_HEADERS} in_memory_search.cc)
add_executable(pq_disk_search ${DISKANN_HEADERS} pq_disk_search.cc pq_utils.cc)
add_executable(in_memory_insert ${DISKANN_HEADERS} in_memory_inserts.cc)
target_link_libraries(in_memory_search ${FASTER_BENCHMARK_LINK_LIBS})
target_link_libraries(pq_disk_search ${FASTER_BENCHMARK_LINK_LIBS})


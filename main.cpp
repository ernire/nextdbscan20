#include <iostream>
#ifdef MPI_ON
#include <mpi.h>
#endif
#ifdef OMP_ON
#include <omp.h>
#endif
#include "nextdbscan.h"

void usage() {
    std::cout << "NextDBSCAN compiled for OpenMP";
#ifdef MPI_ON
    std::cout << ", MPI";
#endif
#ifdef HDF5_ON
    std::cout << ", HDF5";
#endif
#ifdef CUDA_ON
    std::cout << ", CUDA (V100)";
#endif
    std::cout << std::endl << std::endl;
    std::cout << "Usage: [executable] -m minPoints -e epsilon -t threads [input file]" << std::endl;
    std::cout << "    -m minPoints : DBSCAN parameter, minimum number of points required to form a cluster, postive integer, required" << std::endl;
    std::cout << "    -e epsilon   : DBSCAN parameter, maximum neighborhood search radius for cluster, positive floating point, required" << std::endl;
    std::cout << "    -t threads   : Processing parameter, the number of threads to use, positive integer, defaults to number of cores" << std::endl;
    std::cout << "    -o output    : Output file containing the cluster ids in the same order as the input" << std::endl;
    std::cout << "    -h help      : Show this help message" << std::endl << std::endl;
    std::cout << "Supported Input Types:" << std::endl;

    std::cout << ".csv: Text file with one sample/point per line and features/dimensions separated by a space delimiter, i.e. ' '" << std::endl;
    std::cout << ".bin: Custom binary format for faster file reads. Use cvs2bin executable to transform csv into bin files." << std::endl;
#ifdef HDF5_ON
    std::cout << ".hdf5: The best I/O performance when using multiple nodes." << std::endl;
#endif
}

int main(int argc, char** argv) {
    char *p;
    int m = -1;
    float e = -1;
    int n_thread = -1;
    int errors = 0;
    std::string input_file = "";
    std::string output_file = "";

    m = std::stoi(argv[2]);
    e = std::strtof(argv[4], &p);
    n_thread = std::stoi(argv[6]);
    input_file = argv[7];
    std::cout << "input file: " << input_file << " : " << m << " : " << e << " n_threads: " << n_thread << std::endl;

    if (errors || m == -1 || e == -1) {
        std::cout << "Input Error: Please specify the m and e parameters" << std::endl << std::endl;
        usage();
        std::exit(EXIT_FAILURE);
    }
#ifdef MPI_ON
    MPI_Init(&argc, &argv);
#endif
#ifdef OMP_ON
    omp_set_num_threads(n_thread);
#endif
    auto mpi = magmaMPI::build();
    auto results = nextdbscan::start(m, e, n_thread, input_file, mpi);
#ifdef MPI_ON
    MPI_Finalize();
#endif

    if (mpi.rank == 0) {
        std::cout << std::endl;
        std::cout << "Estimated clusters: " << results.clusters << std::endl;
        std::cout << "Core Points: " << results.core_count << std::endl;
        std::cout << "Noise Points: " << results.noise << std::endl;

        /*
        if (output_file.length() > 0) {
            std::cout << "Writing output to " << output_file << std::endl;
            std::ofstream os(output_file);
            // TODO
            for (int i = 0; i < results.n; ++i) {
                os << results.point_clusters[i] << std::endl;
            }
//            for (auto &c : results.point_clusters) {
//                os << c << '\n';
//            }
            os.flush();
            os.close();
            std::cout << "Done!" << std::endl;
        }
         */
    }
    return 0;
}

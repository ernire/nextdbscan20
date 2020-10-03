//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_NEXTDBSCAN_H
#define NEXTDBSCAN20_NEXTDBSCAN_H

#include <string>
#include "data_process.h"
#include "magma_mpi.h"

namespace nextdbscan {

    static const uint8_t NC = 0;
    static const uint8_t AC = 1;
    static const uint8_t SC = 2;

    struct result {
        int clusters;
        int noise;
        int core_count;
        int n;
        // TODO avoid a memory leak
//        int *point_clusters;
    };

    result start(int m, float e, int n_thread, const std::string &in_file, magmaMPI mpi) noexcept;

}


#endif //NEXTDBSCAN20_NEXTDBSCAN_H

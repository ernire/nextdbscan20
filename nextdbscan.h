//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_NEXTDBSCAN_H
#define NEXTDBSCAN20_NEXTDBSCAN_H

#include <string>
#include "nc_tree.h"
#include "magma_mpi.h"

namespace nextdbscan {

    static const uint8_t NC = 0;
    static const uint8_t AC = 1;
    static const uint8_t SC = 2;

    struct result {
        long clusters;
        long noise;
        long core_count;
        long n;
        // TODO avoid a memory leak
        long *point_clusters;
    };

    result start(int m, float e, int n_thread, const std::string &in_file, magmaMPI mpi) noexcept;

}


#endif //NEXTDBSCAN20_NEXTDBSCAN_H

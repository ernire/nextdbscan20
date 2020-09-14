//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_MAGMA_META_H
#define NEXTDBSCAN20_MAGMA_META_H

#include <vector>
#ifdef OMP_ON
#include "magma_exa_omp.h"
#else
#include "magma_exa.h"
#endif

template <typename T>
using s_vec = std::vector<T>;
template <typename T>
using d_vec = std::vector<std::vector<T>>;

#endif //NEXTDBSCAN20_MAGMA_META_H

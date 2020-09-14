//
// Created by Ernir Erlingsson on 19.8.2020.
//
#include "magma_util.h"

int magma_util::get_block_size(int const block_index, int const number_of_samples, int const number_of_blocks) noexcept {
    int block = (number_of_samples / number_of_blocks);
    int reserve = number_of_samples % number_of_blocks;
//    Some processes will need one more sample if the data size does not fit completely with the number of processes
    if (reserve > 0 && block_index < reserve) {
        return block + 1;
    }
    return block;
}

int magma_util::get_block_offset(int const block_index, int const number_of_samples, int const number_of_blocks) noexcept {
    int offset = 0;
    for (int i = 0; i < block_index; i++) {
        offset += get_block_size(i, number_of_samples, number_of_blocks);
    }
    return offset;
}

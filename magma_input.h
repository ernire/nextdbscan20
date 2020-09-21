//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_MAGMA_INPUT_H
#define NEXTDBSCAN20_MAGMA_INPUT_H

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include "magma_meta.h"
#include "magma_util.h"

namespace magma_input {

    void count_lines_and_dimensions(std::string const &in_file, int &lines, int &dim) noexcept {
        std::ifstream is(in_file);
        std::string line, buf;
        int cnt = 0;
        dim = 0;
        while (std::getline(is, line)) {
            if (dim == 0) {
                std::istringstream iss(line);
                std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                        std::istream_iterator<std::string>());
                dim = results.size();
            }
            ++cnt;
        }
        lines = cnt;
        is.close();
    }

    inline bool is_equal(const std::string &in_file, const std::string &s_cmp) noexcept {
        return in_file.compare(in_file.size() - s_cmp.size(), s_cmp.size(), s_cmp) == 0;
    }

    void read_input_bin(const std::string &in_file, s_vec<float> &v_points, int &n_coord, int &n_dim,
            int const n_nodes, int const i_node) noexcept {
        std::ifstream ifs(in_file, std::ios::in | std::ifstream::binary);
        ifs.read((char *) &n_coord, sizeof(int));
        ifs.read((char *) &n_dim, sizeof(int));
        auto size = get_block_size(i_node, n_coord, n_nodes);
        auto offset = get_block_offset(i_node, n_coord, n_nodes);
        auto feature_offset = 2 * sizeof(int) + (offset * n_dim * sizeof(float));
        v_points.resize(size * n_dim);
        ifs.seekg(feature_offset, std::istream::beg);
        ifs.read((char *) &v_points[0], size * n_dim * sizeof(float));
        ifs.close();
    }

    void read_input_csv(const std::string &in_file, s_vec<float> &v_points, long const n_dim) noexcept {
        std::ifstream is(in_file, std::ifstream::in);
        std::string line, buf;
        std::stringstream ss;
        int index = 0;
        while (std::getline(is, line)) {
            ss.str(std::string());
            ss.clear();
            ss << line;
            for (int j = 0; j < n_dim; j++) {
                ss >> buf;
                v_points[index++] = static_cast<float>(atof(buf.c_str()));
            }
        }
        is.close();
    }

    void read_input(const std::string &in_file, s_vec<float> &v_input, int &n, int &n_dim,
            int const n_nodes, int const i_node) noexcept {
        std::string s_cmp_bin = ".bin";
        std::string s_cmp_hdf5_1 = ".h5";
        std::string s_cmp_hdf5_2 = ".hdf5";
        std::string s_cmp_csv = ".csv";

        if (is_equal(in_file, s_cmp_bin)) {
            read_input_bin(in_file, v_input, n, n_dim, n_nodes, i_node);
        } else if (is_equal(in_file, s_cmp_hdf5_1) || is_equal(in_file, s_cmp_hdf5_2)) {

        } else if (is_equal(in_file, s_cmp_csv)) {
            count_lines_and_dimensions(in_file, n, n_dim);
            v_input.resize(n * n_dim);
            std::cout << "WARNING: USING SLOW CSV I/O." << std::endl;
            read_input_csv(in_file, v_input, n_dim);
        }
    }
}



/*
    uint read_input_hdf5(const std::string &in_file, s_vec<float> &v_points, unsigned long &max_d,
            unsigned long const n_nodes, unsigned long const node_index) noexcept {
        uint n = 0;
#ifdef HDF5_ON
        // TODO H5F_ACC_RDONLY ?
        hid_t file = H5Fopen(in_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t dset = H5Dopen1(file, "DBSCAN");
        hid_t fileSpace= H5Dget_space(dset);

        // Read dataset size and calculate chunk size
        hsize_t count[2];
        H5Sget_simple_extent_dims(fileSpace, count,NULL);
        n = count[0];
        max_d = count[1];
        std::cout << "HDF5 total size: " << n << std::endl;

//        hsize_t chunkSize =(this->m_totalSize / this->m_mpiSize) + 1;
//        hsize_t offset[2] = {this->m_mpiRank * chunkSize, 0};
//        count[0] = std::min(chunkSize, this->m_totalSize - offset[0]);
//        uint deep_io::get_block_size(const uint block_index, const uint number_of_samples, const uint number_of_blocks) {

        hsize_t block_size =  deep_io::get_block_size(node_index, n, n_nodes);
        hsize_t block_offset =  deep_io::get_block_start_offset(node_index, n, n_nodes);
        hsize_t offset[2] = {block_offset, 0};
        count[0] = block_size;
        v_points.resize(block_size * max_d);

        hid_t memSpace = H5Screate_simple(2, count, NULL);
        H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET,offset, NULL, count, NULL);
        H5Dread(dset, H5T_IEEE_F32LE, memSpace, fileSpace,H5P_DEFAULT, &v_points[0]);

        H5Dclose(dset);
        H5Fclose(file);
#endif
#ifndef HDF5_ON
        std::cerr << "Error: HDF5 is not supported by this executable. "
                     "Use the cu-hdf5 flag when building from source to support HDF5." << std::endl;
        exit(-1);
#endif
        return n;
    }

 */

#endif //NEXTDBSCAN20_MAGMA_INPUT_H

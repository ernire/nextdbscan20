//
// Created by Ernir Erlingsson on 16.8.2020.
//

#ifndef NEXTDBSCAN20_MAGMA_MPI_H
#define NEXTDBSCAN20_MAGMA_MPI_H

#ifdef MPI_ON
#include <mpi.h>
#endif
#include <cassert>

class magmaMPI {
private:

#ifdef MPI_ON
    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    int inferType(std::vector<T> &v) noexcept {
        if (std::is_floating_point<T>::value) {
            return MPI_FLOAT;
        } else {
            return MPI_INT;
        }
    }
#endif

    explicit magmaMPI(int const mpi_rank, int const mpi_comm, int const n_nodes) :
            rank(mpi_rank), comm(mpi_comm), n_nodes(n_nodes) {
#if defined(DEBUG_ON) && defined(MPI_ON)
        assert(n_nodes > 0);
        // TODO support dynamic comm size
        assert(mpi_comm == MPI_COMM_WORLD);
        int size;
        MPI_Comm_size(mpi_comm, &size);
        assert(n_nodes == size);
#endif
    }

public:
    int const rank, comm, n_nodes;

    enum Op { undefined, max, min, sum };

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allGather(d_vec<T> &v_buf) {
#ifdef MPI_ON
        auto type = inferType(v_buf);
        MPI_Allgather(MPI_IN_PLACE,
                0, // ignored
                type, // ignored
                &v_buf[0],
                v_buf.size() / n_nodes,
                type,
                comm);
#endif
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allGatherv(d_vec<T> &v_buf, d_vec<T> &v_size, d_vec<T> &v_offset) {
#ifdef MPI_ON
        auto type = inferType(v_buf);
        MPI_Allgatherv(MPI_IN_PLACE,
                0,
                type,
                &v_buf[0],
                &v_size[0],
                &v_offset[0],
                type,
                comm);
#endif
    }

    template<typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    void allReduce(d_vec<T> &v_buf, Op op) noexcept {
#ifdef MPI_ON
        int iOp = undefined;
        switch (op) {
            case max: iOp = MPI_MAX; break;
            case min: iOp = MPI_MIN; break;
            case sum: iOp = MPI_SUM; break;
            case undefined:
                return;
        }
#ifdef DEBUG_ON
        assert(iOp != undefined);
#endif
        MPI_Allreduce(MPI_IN_PLACE,
                &v_buf[0],
                static_cast<int>(v_buf.size()),
                inferType(v_buf),
                iOp,
                comm);
#endif
    }

    static magmaMPI build() {
#ifdef MPI_ON
        int mpi_size;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        return *new magmaMPI(mpi_rank, MPI_COMM_WORLD, mpi_size);
#else
        return *new magmaMPI(0, 0, 1);
#endif

    }
};

#endif //NEXTDBSCAN20_MAGMA_MPI_H

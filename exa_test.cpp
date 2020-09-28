/*
Copyright (c) 2019, Ernir Erlingsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */
#include "gtest/gtest.h"
#include "magma_meta.h"
#include <algorithm>
#include <numeric>
#include <random>

TEST(exa, fill) {
    s_vec<int> v_test(1000, 0);
    exa::fill(v_test, 100, 200, 1);
    for (int i = 0; i < v_test.size(); ++i) {
        if (i < 100 || i >= 200) {
            EXPECT_EQ(v_test[i], 0);
        } else {
            EXPECT_EQ(v_test[i], 1);
        }
    }
}

TEST(exa, iota) {
    s_vec<int> v_test(1000);
    exa::iota(v_test, 0, v_test.size(), 0);
    for (int i = 0; i < v_test.size(); ++i) {
        EXPECT_EQ(v_test[i], i);
    }
    exa::iota(v_test, 100, 200, 5000);
    EXPECT_EQ(v_test[100], 5000);
    EXPECT_EQ(v_test[199], 5099);
    EXPECT_EQ(v_test[200], 200);
}

TEST(exa, sort) {
    s_vec<int> v_test(1000);
    std::iota(v_test.begin(), v_test.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v_test.begin(), v_test.end(), g);
    exa::sort(v_test, 0, v_test.size(), [](int const &v1, int const &v2) -> bool {
        return v1 > v2;
    });
    EXPECT_EQ(v_test[999], 0);
    EXPECT_EQ(v_test[0], 999);
    EXPECT_EQ(v_test[500], 499);

    // TODO partial sort
}

TEST(exa, transform) {
    s_vec<int> v_input(1000);
    s_vec<int> v_output(1000);

    exa::transform(v_input, v_output, 0, v_input.size(), 0, [](int const &v) -> int {
        return 1;
    });

    for (int const &v : v_output) {
        EXPECT_EQ(v, 1);
    }
    for (int const &v : v_input) {
        EXPECT_EQ(v, 0);
    }

    // TODO much more
}

TEST(exa, count_if) {
    s_vec<int> v_test(1000);
    exa::fill(v_test, 0, v_test.size(), 1);

    auto cnt = exa::count_if(v_test, 0, v_test.size(), [](int const &v) -> bool {
        return true;
    });
    EXPECT_EQ(cnt, 1000);
}

TEST(exa, copy_if) {
    s_vec<int> v_input(10);
    s_vec<int> v_output(10);
    exa::iota(v_input, 0, v_input.size(), 0);
    exa::copy_if(v_input, v_output, 0, v_input.size(), 0, [](auto const val) -> bool {
        return val % 2 == 0;
    });
    /*
    s_vec<int> v_input(1000);
    s_vec<int> v_output(1000);

    exa::iota(v_input, 0, v_input.size(), 0);

    exa::copy_if(v_input, v_output, 0, v_input.size(), 0, [](auto const val) -> bool {
       return val % 2 == 0;
    });
    EXPECT_EQ(v_output.size(), 500);
    for (int i = 1; i < v_output.size(); ++i) {
        EXPECT_TRUE(v_output[i-1] < v_output[i] && v_output[i] % 2 == 0);
    }
     */
}

TEST(exa, reduce) {
    s_vec<int> v_test(1000);
    exa::fill(v_test, 0, v_test.size(), 1);

    auto sum = exa::reduce(v_test, 0, v_test.size(), 1);
    EXPECT_EQ(sum, 1001);
}

TEST(exa, for_each) {
    s_vec<int> v_test(1000, 0);
    exa::for_each(0, v_test.size(), [&](auto i) -> void {
        v_test[i] = 1;
    });
    auto sum = exa::reduce(v_test, 0, v_test.size(), 0);
    EXPECT_EQ(sum, v_test.size());
}

TEST(exa, exclusive_scan) {
    s_vec<int> v_input(1000, 1);
    s_vec<int> v_output(1000);
    exa::exclusive_scan(v_input, v_output, 0, v_input.size(), 0, 100);
    EXPECT_EQ(v_output[0], 100);
    EXPECT_EQ(v_output[100], 200);
    EXPECT_EQ(v_output[v_output.size()-1], 1099);
}

TEST(exa, min_max_element) {
    s_vec<int> v_input(1000, 1);
    v_input[33] = -100;
    v_input[777] = 100;
    auto pair = exa::minmax_element(v_input, 0, v_input.size(), [&](auto const v1, auto const v2) -> bool {
        return v1 < v2;
    });
    EXPECT_EQ(pair.first, -100);
    EXPECT_EQ(pair.second, 100);
    pair = exa::minmax_element(v_input, 50, v_input.size(), [&](auto const v1, auto const v2) -> bool {
        return v1 < v2;
    });
    EXPECT_EQ(pair.first, 1);
    EXPECT_EQ(pair.second, 100);
}

int main(int argc, char **argv) {
    std::cout << "STARTING TESTS" << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

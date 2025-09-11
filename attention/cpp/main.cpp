#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <algorithm>


class Matrix {
    private:
        std::vector<std::vector<double>> data_;
        size_t rows_, cols_;


    public:
        Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols){
            data_.resize(rows, std::vector<double>(cols, 0.0));


            // will be back here soon I am thinking for now if we should do something more like
            // calculating the index of the data or not

        }

        double& operator()(size_t row, size_t col) { return data_[row][col]; }
        const double& operator()(size_t row, size_t col) const { return data_[row][col]; }



};
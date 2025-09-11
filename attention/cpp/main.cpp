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

        size_t rows() const { return rows_;}
        size_t cols() const { return cols_;}


        // performing matrix multiplication
        Matrix matmul(const Matrix& other) const {
            if (cols_ != other.rows_) throw std::invalid_argument("Matrix dimension mismatch");
            Matrix result(rows_, other.cols_);
            for (size_t i = 0; i < rows_; ++i){
                for (size_t j = 0; j < other.cols_; ++j) {
                    for (size_t k = 0; k < cols_; ++k) {
                        result(i, j) += data_[i][k] * other(k, j);
                    }
                }
            }

            return result;
        }


        Matrix transpose() const {


            Matrix result(cols_, rows_);

            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j){
                    

                    // here we will be mirroring the result 
                    // with turned down numbers like the actual transpose would look like

                    result(j, i) = data_[i][j];
                }
            }

            // thinking hard on the transpose part
            // its a little bit tricky for me T _ T 
            return result;
        }



        void scale(double scalar){
            for (size_t i = 0; i < rows_; ++i){
                for (size_t j = 0; j < cols_; ++j){
                    data_[i][j] *= scalar;
                }
            }
        }
};
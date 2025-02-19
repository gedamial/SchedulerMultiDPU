#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "datatypes.h"
#include <iostream>

float computeTd(const Matrix<bool>& S, const size_t dpu_idx);

float computeTtot(const Matrix<bool>& S);

float computeIdleTime(const Matrix<bool>& S, const size_t dpu_idx);

float computeIdleEnergy(const Matrix<bool>& S, const size_t dpu_idx);

float computeEd(const Matrix<bool>& S, const size_t dpu_idx);

float computeEtot(const Matrix<bool>& S);

std::vector<std::string> Explode(std::string str, std::string delim);

template<typename T>
inline void PrintMatrix(const Matrix<T>& mat, std::ostream& os)
{
    for (const auto& row : mat)
    {
        for (const auto& col : row)
            os << col << " ";
        os << std::endl;
    }

    os << std::endl;
}

template<typename T>
inline void PrintVector(const std::vector<T>& vec, std::ostream& os)
{
    for (const T& elem : vec)
        os << elem << " ";

    os << std::endl;
}

#endif
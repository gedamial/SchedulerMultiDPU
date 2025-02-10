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

inline void PrintMatrix(const Matrix<bool>& mat)
{
    for (const auto& row : mat)
    {
        for (const auto& col : row)
            std::cout << col << " ";
        std::cout << std::endl;
    }
}

#endif
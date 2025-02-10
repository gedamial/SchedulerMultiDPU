#include "../inc/functions.h"
#include "../inc/datatypes.h"

#include <algorithm>
#include <iostream>

extern float B0;
extern float B1;
extern Map t;
extern Map p;
extern std::vector<std::string> DPUs;
extern std::vector<std::string> workloads;

float computeTd(const Matrix<bool>& S, const size_t dpu_idx)
{
    float Td = 0;

    for(int i = 0; i < workloads.size(); ++i)
        if(S[dpu_idx][i] == 1)
        {
            Td += t.at({DPUs[dpu_idx], workloads[i]});
        }

    size_t numWorkloadsAssigned = std::count_if(S[dpu_idx].begin(), S[dpu_idx].end(), [](const bool b) { return b;});

    if (numWorkloadsAssigned > 1)
        Td *= B0 / numWorkloadsAssigned + B1;

    return Td;
}

float computeTtot(const Matrix<bool>& S)
{
    float Ttot = 0;

    for(int i = 0; i < S.size(); ++i)
    {
        const float Td = computeTd(S,i);

        if (Td > Ttot)
            Ttot = Td;
    }

    return Ttot;
}

float computeIdleTime(const Matrix<bool>& S, const size_t dpu_idx)
{
    return computeTtot(S)-computeTd(S, dpu_idx);
}

float computeIdleEnergy(const Matrix<bool>& S, const size_t dpu_idx)
{
    return p.at({DPUs[dpu_idx], "Idle"})*computeIdleTime(S, dpu_idx);
}

float computeEd(const Matrix<bool>& S, const size_t dpu_idx)
{
    float Ed = 0;

    for(int i = 0; i < workloads.size(); ++i)
    {
        if(S[dpu_idx][i] == 1)
        {
            Ed += p.at({DPUs[dpu_idx], workloads[i]})*t.at({DPUs[dpu_idx], workloads[i]});
        }
    }

    Ed += computeIdleEnergy(S, dpu_idx);

    size_t numWorkloadsAssigned = std::count_if(S[dpu_idx].begin(), S[dpu_idx].end(), [](const bool b) { return b;});

    if (numWorkloadsAssigned > 1)
        Ed *= B0 / numWorkloadsAssigned + B1;

    return Ed;
}

float computeEtot(const Matrix<bool>& S)
{
    float Etot = 0;

    for(int i = 0; i < S.size(); ++i)
    {
        Etot += computeEd(S, i);
    }

    return Etot;
}

std::vector<std::string> Explode(std::string str, std::string delim)
{
    std::vector<std::string> result;
    size_t pos;

    while((pos = str.find(delim)) != std::string::npos)
    {
        result.push_back(str.substr(0, pos));
        str.erase(0, pos+delim.length());
    }

    result.push_back(str);

    return result;
}
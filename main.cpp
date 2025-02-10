#include <iostream>
#include <unordered_map>
#include <vector>
#include <climits>
#include <fstream>
#include <algorithm>
#include <random>
#include <format>
#include <iomanip>
#include <chrono>

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1,T2> &p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use something like boost.hash_combine
        return h1 ^ h2;
    }
};

// For mapping (a,m) pairs to some float
using Map = std::unordered_map<std::pair<std::string, std::string>, float, pair_hash>;

template<typename T>
using Matrix = std::vector<std::vector<T>>;

// t(a,m)
Map t;
// p(a,m)
Map p;

// Linear regression parameters for k(n)
constexpr float B0 = 0.23f;
constexpr float B1 = 0.72f;

std::vector<std::string> DPUs;
std::vector<std::string> workloads;

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
        const float Td = computeTd(S, i);

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

inline void PrintMatrix(const Matrix<bool>& mat)
{
    for (const auto& row : mat)
    {
        for (const auto& col : row)
            std::cout << col << " ";
        std::cout << std::endl;
    }
}

// argv[0]
// argv[1] DPU file
// argv[2] Workload file
// argv[3] Repetitions
int main(int argc, char** argv)
{
    if(argc != 4)
    {
        std::cout << "Not enough or too many input parameters.\n"
                     "Please provide:\n"
                     "- DPU file\n"
                     "- Workload file\n"
                     "- Number of repetitions" << std::endl;
        return -1;
    }

    std::string line;

    // INPUT: DPU file
    std::ifstream dpu_file(argv[1]);

    if(!dpu_file)
    {
        std::cout << "Could not open DPU file." << std::endl;
        return -1;
    }

    std::cout << "Reading DPU architectures..." << std::endl;
    while(dpu_file >> line)
    {
        std::cout << line << std::endl;
        DPUs.push_back(line);
    }

    // INPUT: Workload file
    std::ifstream workload_file(argv[2]);

    if(!workload_file)
    {
        std::cout << "Could not open Workload file." << std::endl;
        return -1;
    }

    std::cout << "Reading workloads..." << std::endl;
    while(workload_file >> line)
    {
        std::cout << line << std::endl;
        workloads.push_back(line);
    }

    const size_t numDPUs = DPUs.size();
    const size_t numWorkloads = workloads.size();

    // INPUT: t(a,m) MAP
    std::ifstream t_file("../runtimes.csv");

    if(!t_file)
    {
        std::cout << "Could not open t(a,m) file." << std::endl;
        return -1;
    }

    std::getline(t_file, line); // skip first line of .csv file

    std::cout << "Reading runtimes t(a,m)..." << std::endl;
    while(t_file >> line)
    {
        // Explode line "Arch;Model;Runtime"
        std::vector<std::string> tokens = Explode(line, ";");
        t[{tokens[0], tokens[1]}] = std::stof(tokens[2]);
    }

    // INPUT: p(a,m) MAP
    std::ifstream p_file("../avg_power.csv");

    if(!p_file)
    {
        std::cout << "Could not open p(a,m) file." << std::endl;
        return -1;
    }

    std::getline(p_file, line); // skip first line of .csv file

    std::cout << "Reading avg power p(a,m)..." << std::endl;
    while(p_file >> line)
    {
        // Explode line "Arch;Model;Power PS"
        std::vector<std::string> tokens = Explode(line, ";");
        p[{tokens[0], tokens[1]}] = std::stof(tokens[2]);
    }

    const size_t repetitions = atoi(argv[3]);

    // OUTPUT FILES
    std::ofstream output_file_total("schedule_output_total.txt", std::ios::out | std::ios::trunc);
    std::ofstream output_file_dpu("schedule_output_dpu.txt", std::ios::out | std::ios::trunc);
    std::ofstream output_file_idleEnergy("schedule_output_idleEnergy.txt", std::ios::out | std::ios::trunc);

    for(int i = 0; i < repetitions; ++i)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        // BEGIN scheduling

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(workloads.begin(), workloads.end(), g);

        std::cout << "Workloads array after shuffling:" << std::endl;
        for(int i = 0; i < numWorkloads; ++i)
            std::cout << workloads[i] << " ";
        std::cout << std::endl;

        Matrix<bool> S;
        S.resize(numDPUs);
        for(auto& row : S)
            row.resize(numWorkloads);

        for(int i = 0; i < numWorkloads; ++i)
        {
            size_t bestDPU_idx = 0;
            float bestTtot = (float)INT_MAX;

            for(int j = 0; j < numDPUs; ++j)
            {
                S[j][i] = 1;

                float newTtot = computeTtot(S);

                if (newTtot < bestTtot)
                {
                    bestTtot = newTtot;
                    bestDPU_idx = j;
                }

                S[j][i] = 0;
            }

            S[bestDPU_idx][i] = 1;
        }

        // END scheduling

        auto t2 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1);

        std::cout << "Scheduler runtime: " << us.count() << "us" << std::endl;

        PrintMatrix(S);

        // Print runtime and energy for each DPU
        for(int i = 0; i < numDPUs; ++i)
        {
            const float Td = computeTd(S,i);
            const float Ed = computeEd(S,i);
            const float IdleEnergy = computeIdleEnergy(S,i);

            std::cout << std::format("DPU{} \t Td = {:.3f} \t Ed = {:.3f}", i, Td, Ed) << std::endl;
            output_file_dpu << std::fixed << std::setprecision(3) << Td << "," << Ed << ",";
            output_file_idleEnergy << std::fixed << std::setprecision(3) << IdleEnergy << ",";
        }
        output_file_dpu << "\n";
        output_file_idleEnergy << "\n";

        const float Ttot = computeTtot(S);
        const float Etot = computeEtot(S);

        std::cout << "Ttot = " << Ttot << "s" << std::endl;
        std::cout << "Etot = " << Etot << "mJ" <<  std::endl;

        output_file_total << std::fixed << std::setprecision(3) << Ttot << "," << Etot << "\n";
    }

    return 0;
}

/*

FOR MATLAB (readmatrix)

****************************************
OUTPUT FILE schedule_output_total.txt
- A row for each repetition, where values are comma-separated.

Example with n=3 repetitions:

Ttot1,Etot1
Ttot2,Etot2
Ttot3,Etot3
****************************************
****************************************
OUTPUT FILE schedule_output_dpu.txt
- A row for each repetition, where DPU values are comma-separated.

Example with n=3 repetitions and 4 DPUs:

Td0,Ed0,Td1,Ed1,Td2,Ed2,Td3,Ed3
Td0,Ed0,Td1,Ed1,Td2,Ed2,Td3,Ed3
Td0,Ed0,Td1,Ed1,Td2,Ed2,Td3,Ed3
****************************************
****************************************
OUTPUT FILE schedule_output_idleEnergy.txt
- A row for each repetition, where DPU values are comma-separated.

Example with n=3 repetitions and 4 DPUs:

IdleEnergy0,IdleEnergy1,IdleEnergy2,IdleEnergy3
IdleEnergy0,IdleEnergy1,IdleEnergy2,IdleEnergy3
IdleEnergy0,IdleEnergy1,IdleEnergy2,IdleEnergy3
****************************************
*/

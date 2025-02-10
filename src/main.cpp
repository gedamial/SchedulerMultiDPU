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
#include "../inc/datatypes.h"
#include "../inc/functions.h"

// t(a,m)
Map t;
// p(a,m)
Map p;

// Linear regression parameters for k(n)
float B0 = 0.23f;
float B1 = 0.72f;

std::vector<std::string> DPUs;
std::vector<std::string> workloads;

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
    std::ifstream t_file("../files/input/runtimes.csv");

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
    std::ifstream p_file("../files/input/avg_power.csv");

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
    std::ofstream output_file_total("../files/output/schedule_output_total.txt", std::ios::out | std::ios::trunc);
    std::ofstream output_file_dpu("../files/output/schedule_output_dpu.txt", std::ios::out | std::ios::trunc);
    std::ofstream output_file_idleEnergy("../files/output/schedule_output_idleEnergy.txt", std::ios::out | std::ios::trunc);

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

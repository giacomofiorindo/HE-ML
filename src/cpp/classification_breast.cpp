#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <tfhe/tfhe.h>
#include "tfhe/tfhe_garbage_collector.h"
#include <omp.h>
#include <chrono>

// parameters from 'Fast Homomorphic Evaluation of Deep Discretized Neural Networks'
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_n  600                   ///  LweParams
#define SEC_PARAMS_N 1024                   /// TLweParams
#define SEC_PARAMS_k    1                   /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -36)   /// TLweParams
#define SEC_PARAMS_BK_BASEBITS 10           /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
#define SEC_PARAMS_KS_STDDEV pow(2., -25)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH   18           /// Key Switching Params

TFheGateBootstrappingParameterSet *get_bootstrapping_parameters(int minimum_lambda) {
        if (minimum_lambda > 128)
                std::cerr << "Sorry, for now, the parameters are only implemented for about 128bit of security!\n";

        static const int n = SEC_PARAMS_n;
        static const int N = SEC_PARAMS_N;
        static const int k = SEC_PARAMS_k;
        static const double max_stdev = SEC_PARAMS_STDDEV;

        static const int bk_Bgbit = SEC_PARAMS_BK_BASEBITS; //<-- ld, thus: 2^10
        static const int bk_l = SEC_PARAMS_BK_LENGTH;
        static const double bk_stdev = SEC_PARAMS_BK_STDDEV;

        static const int ks_basebit = SEC_PARAMS_KS_BASEBITS; //<-- ld, thus: 2^1
        static const int ks_length = SEC_PARAMS_KS_LENGTH;
        static const double ks_stdev = SEC_PARAMS_KS_STDDEV;


        LweParams *params_in = new_LweParams(n, ks_stdev, max_stdev);
        TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
        TGswParams *params_bk = new_TGswParams(bk_l, bk_Bgbit, params_accum);

        TfheGarbageCollector::register_param(params_in);
        TfheGarbageCollector::register_param(params_accum);
        TfheGarbageCollector::register_param(params_bk);

        return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

template<typename T, typename A>
inline void extract_array(std::string filename, std::vector<T, A> &vec) {
        std::string line;
        std::ifstream fin(filename);

        if (fin.is_open()) {
                while (getline(fin, line)) {
                        std::stringstream ss(line);

                        while (getline(ss, line, ',')) {
                                vec.push_back(std::stoi(line));
                        }
                }
        }
}

int main(int argc, char *argv[]) {
        int i = 0;
        std::vector<int> input;
        std::vector<int> labels;
        std::vector<std::vector<int> > weights;
        std::vector<std::vector<int> > biases;

        for (int i = 0; i < 3; ++i) {
                weights.push_back(std::vector<int>());
                biases.push_back(std::vector<int>());
        }

        extract_array("../../resources/breast/weights_and_biases/q_breast_biases0.csv", biases[0]);
        extract_array("../../resources/breast/weights_and_biases/q_breast_biases1.csv", biases[1]);
        extract_array("../../resources/breast/weights_and_biases/q_breast_biases2.csv", biases[2]);
        extract_array("../../resources/breast/weights_and_biases/q_breast_weights0.csv", weights[0]);
        extract_array("../../resources/breast/weights_and_biases/q_breast_weights1.csv", weights[1]);
        extract_array("../../resources/breast/weights_and_biases/q_breast_weights2.csv", weights[2]);
        
        extract_array("../../resources/breast/input_breast_test.csv", input);
        extract_array("../../resources/breast/labels_breast_test.csv", labels);
        
        int num_threads;

        std::cout << "done" << std::endl;

        TFheGateBootstrappingParameterSet *params = get_bootstrapping_parameters(80);
        TFheGateBootstrappingSecretKeySet *secret_key = new_random_gate_bootstrapping_secret_keyset(params);

        double alpha = pow(2., -20);
        int number_input_neurons;
        int number_output_neurons;
        std::vector<int> bias_layer;
        std::vector<int> weights_layer;
        int16_t b, w, limit, start;
        Torus32 mu;
        int bs_space = 300;
        const Torus32 bs_mu = modSwitchToTorus32(1, bs_space); // TODO: use a variable for size
        int message_space = 300;
        LweSample *multi_sum, *bootstrapped_multi_sum;
        int layers = 4;
        int neurons[4] = {30, 40, 20, 2};
        double correct, total = 0;

        LweSample *enc_input = new_LweSample_array(30, secret_key->params->in_out_params);
        LweSample *result = new_LweSample_array(2, secret_key->params->in_out_params);

        const LweParams *in_out_params = secret_key->cloud.params->in_out_params;
        const LweBootstrappingKeyFFT *bs_key = secret_key->cloud.bkFFT;

        int16_t size = 0;
        for (int i = 0; i < layers; ++i) {
                if (neurons[i] > size)
                        size = neurons[i];
        }

        if (argc < 2) num_threads = omp_get_num_threads();
        else num_threads = std::stoi(argv[1]);
        if (argc < 3) limit = 114;
        else limit = std::stoi(argv[2]);
        if (argc < 4) start = 0;
        else start = std::stoi(argv[3]);

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int k = start; k < start + limit; ++k) {

                multi_sum = new_LweSample_array(size, in_out_params);
                bootstrapped_multi_sum = new_LweSample_array(size, in_out_params);

                for (int i = 0; i < 30; ++i) {
                        mu = modSwitchToTorus32(input[k * 30 + i], message_space);
                        lweSymEncrypt(bootstrapped_multi_sum + i, mu, alpha, secret_key->lwe_key);
                }

                // FIRST LAYERS
                for (int l = 0; l < layers - 1; ++l) {
                        number_input_neurons = neurons[l];
                        number_output_neurons = neurons[l + 1];
                        bias_layer = biases[l];
                        weights_layer = weights[l];

                        // Calculate multi sum per each output neuron
                        #pragma omp parallel for num_threads(num_threads)
                        for (int o = 0; o < number_output_neurons; ++o) {
                                b = bias_layer[o];
                                mu = modSwitchToTorus32(b, message_space);
                                lweNoiselessTrivial(multi_sum + o, mu, in_out_params); //Add bias without noise -> (0, bias)

                                int n = o * number_input_neurons;
                                for (int i = 0; i < number_input_neurons; ++i) {
                                        w = weights_layer[n + i];
                                        lweAddMulTo(multi_sum + o, w, bootstrapped_multi_sum + i, in_out_params);
                                }
                        }


                        if (l < layers - 2) {
                                message_space = bs_space;
                                #pragma omp parallel for num_threads(num_threads)
                                for (int o = 0; o < number_output_neurons; ++o) {
                                        // Bootstrap
                                        tfhe_bootstrap_FFT(bootstrapped_multi_sum + o, bs_key, bs_mu, multi_sum + o);
                                }
                        }

                        if (l == layers - 2) {
                                #pragma omp parallel for num_threads(num_threads)
                                for (int i = 0; i < number_output_neurons; ++i) {
                                        lweCopy(result + i, multi_sum + i, in_out_params);
                                }
                        }
                }

                int correct_label = -1;
                int max_score = 0;

                for (int i = 0; i < 2; ++i) {
                        int score = lwePhase(result + i, secret_key->lwe_key);
                        if (score > max_score) {
                                max_score = score;
                                correct_label = i;
                        }
                }

                if (correct_label == labels[k])
                        ++correct;
                ++total;

                delete_LweSample_array(size, multi_sum);
                delete_LweSample_array(size, bootstrapped_multi_sum);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto duration_sc = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        auto duration_min = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
        std::cout << "######TIMING######" << std::endl;
        std::cout << duration_ms.count() << " ms" << std::endl;
        std::cout << duration_sc.count() << " sec" << std::endl;
        std::cout << duration_min.count() << " min" << std::endl;

        double res = correct / total;
        std::cout << "accuracy: " << res << std::endl;

        delete_LweSample_array(30, enc_input);
        delete_LweSample_array(2, result);

        delete_gate_bootstrapping_secret_keyset(secret_key);
        delete_gate_bootstrapping_parameters(params);

        return 0;
}

#ifndef SQ_SCAN_H
#define SQ_SCAN_H

#include <vector>
#include <queue>
#include <utility>
#include <cstdint>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath> 
#include <iostream> 

struct SQQuantizer {
    std::vector<float> min_values;
    std::vector<float> scales;
    std::vector<float> inv_scales; 
    size_t dim_;

    SQQuantizer(size_t dim) : dim_(dim) {
        min_values.resize(dim);
        scales.resize(dim);
        inv_scales.resize(dim);
    }
    //训练量化器
    void train(const float* data, size_t n, size_t dim) {
        if (dim != dim_) {
             std::cerr << "Error: Training dimension mismatch. Expected " << dim_ << ", got " << dim << std::endl;
             return; 
        }
        std::vector<float> max_values(dim, std::numeric_limits<float>::lowest());
        std::fill(min_values.begin(), min_values.end(), std::numeric_limits<float>::max());

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < dim; j++) {
                float val = data[i * dim + j];
                min_values[j] = std::min(min_values[j], val);
                max_values[j] = std::max(max_values[j], val);
            }
        }

        for (size_t j = 0; j < dim; j++) {
            float range = max_values[j] - min_values[j];
            if (range > 1e-9f) { 
                scales[j] = 255.0f / range;
                inv_scales[j] = range / 255.0f; 
            } else {
                scales[j] = 0.0f;
                inv_scales[j] = 0.0f; //当范围很小时，认为 scale 为 0，inv_scale 也为 0
            }
        }
    }

    //量化向量
    void quantize(const float* input, uint8_t* output, size_t n, size_t dim) {
         if (dim != dim_) {
              std::cerr << "Error: Quantization dimension mismatch. Expected " << dim_ << ", got " << dim << std::endl;
              return;
         }
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < dim; j++) {
                 float normalized = 0.0f;
                 if (std::abs(scales[j]) > 1e-9f) {
                     normalized = (input[i * dim + j] - min_values[j]) * scales[j];
                 }
                 if (normalized < 0.0f) normalized = 0.0f;
                 if (normalized > 255.0f) normalized = 255.0f;
                 //四舍五入到最近的整数
                 output[i * dim + j] = static_cast<uint8_t>(normalized + 0.5f);
            }
        }
    }

#ifdef __ARM_NEON
    float compute_ip(const uint8_t* quantized, const float* query, size_t dim) const {
        if (dim != dim_) {
            std::cerr << "Error: compute_ip dimension mismatch. Expected " << dim_ << ", got " << dim << std::endl;
            return 0.0f; 
        }
        
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t i = 0;
        const size_t step = 16; 

        for (; i + (step - 1) < dim; i += step) {
             uint8x16_t q_u8 = vld1q_u8(quantized + i);

             uint16x8_t q_u16_low = vmovl_u8(vget_low_u8(q_u8));
             uint16x8_t q_u16_high = vmovl_u8(vget_high_u8(q_u8));

             uint32x4_t q_u32_0 = vmovl_u16(vget_low_u16(q_u16_low));
             uint32x4_t q_u32_1 = vmovl_u16(vget_high_u16(q_u16_low));
             uint32x4_t q_u32_2 = vmovl_u16(vget_low_u16(q_u16_high));
             uint32x4_t q_u32_3 = vmovl_u16(vget_high_u16(q_u16_high));

             float32x4_t q_f32_0 = vcvtq_f32_u32(q_u32_0);
             float32x4_t q_f32_1 = vcvtq_f32_u32(q_u32_1);
             float32x4_t q_f32_2 = vcvtq_f32_u32(q_u32_2);
             float32x4_t q_f32_3 = vcvtq_f32_u32(q_u32_3);

             float32x4_t mins_0 = vld1q_f32(min_values.data() + i);
             float32x4_t inv_sc_0 = vld1q_f32(inv_scales.data() + i);
             float32x4_t mins_1 = vld1q_f32(min_values.data() + i + 4);
             float32x4_t inv_sc_1 = vld1q_f32(inv_scales.data() + i + 4);
             float32x4_t mins_2 = vld1q_f32(min_values.data() + i + 8);
             float32x4_t inv_sc_2 = vld1q_f32(inv_scales.data() + i + 8);
             float32x4_t mins_3 = vld1q_f32(min_values.data() + i + 12);
             float32x4_t inv_sc_3 = vld1q_f32(inv_scales.data() + i + 12);

             q_f32_0 = vfmaq_f32(mins_0, q_f32_0, inv_sc_0);
             q_f32_1 = vfmaq_f32(mins_1, q_f32_1, inv_sc_1);
             q_f32_2 = vfmaq_f32(mins_2, q_f32_2, inv_sc_2);
             q_f32_3 = vfmaq_f32(mins_3, q_f32_3, inv_sc_3);

             float32x4_t query_0 = vld1q_f32(query + i);
             float32x4_t query_1 = vld1q_f32(query + i + 4);
             float32x4_t query_2 = vld1q_f32(query + i + 8);
             float32x4_t query_3 = vld1q_f32(query + i + 12);

             sum_vec = vfmaq_f32(sum_vec, q_f32_0, query_0);
             sum_vec = vfmaq_f32(sum_vec, q_f32_1, query_1);
             sum_vec = vfmaq_f32(sum_vec, q_f32_2, query_2);
             sum_vec = vfmaq_f32(sum_vec, q_f32_3, query_3);
        }

        //水平求和NEON累加的部分
        float sum = 0.0f;
        #if defined(__aarch64__)
        sum = vaddvq_f32(sum_vec);
        #else 
        float32x2_t sum_p = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        sum_p = vpadd_f32(sum_p, sum_p);
        sum = vget_lane_f32(sum_p, 0);
        #endif

        for (; i < dim; i++) {
             float inv_scale = inv_scales[i];
             float reconstructed_val = min_values[i]; 
             if (std::abs(inv_scale) > 1e-20f) { 
                 reconstructed_val = static_cast<float>(quantized[i]) * inv_scale + min_values[i];
             }
             sum += reconstructed_val * query[i];
        }

        return sum;
    }
#else
    float compute_ip(const uint8_t* quantized, const float* query, size_t dim) const {
        if (dim != dim_) {
            std::cerr << "Error: compute_ip dimension mismatch. Expected " << dim_ << ", got " << dim << std::endl;
            return 0.0f; 
        }
        
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float inv_scale = inv_scales[i];
            float reconstructed_val = min_values[i]; 
            if (std::abs(inv_scale) > 1e-20f) { 
                 reconstructed_val = static_cast<float>(quantized[i]) * inv_scale + min_values[i];
             }
            sum += reconstructed_val * query[i];
        }
        return sum;
    }
#endif 

}; 

std::priority_queue<std::pair<float, uint32_t>> sq_search(
    const uint8_t* base_quant,     
    const SQQuantizer& quantizer, 
    const float* query,           
    size_t base_number,           
    size_t dim,                    
    size_t k                      
) {
    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>> top_k_heap; 
    for (size_t i = 0; i < base_number; ++i) {
        const uint8_t* current_base_quant = base_quant + i * dim;

        float ip = quantizer.compute_ip(current_base_quant, query, dim);

        if (top_k_heap.size() < k) {
            top_k_heap.push({ip, static_cast<uint32_t>(i)});
        } else if (ip > top_k_heap.top().first) {
            top_k_heap.pop();
            top_k_heap.push({ip, static_cast<uint32_t>(i)});
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> result_max_heap;
    while(!top_k_heap.empty()) {
        result_max_heap.push(top_k_heap.top());
        top_k_heap.pop();
    }

    return result_max_heap; 
}
#endif 

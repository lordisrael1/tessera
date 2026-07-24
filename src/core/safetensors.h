#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Reader for the safetensors format:
//   [8 bytes little-endian u64 = header length N]
//   [N bytes of JSON: name -> {dtype, shape, data_offsets:[begin,end]}]
//   [raw tensor blob]
// The file is mmap'd PROT_READ once; every TensorInfo points into that mapping.
// Loading a 1 GB model should take milliseconds, not seconds — if it doesn't,
// you copied when you should have mapped.

enum class DType { F32, F16, BF16, I64, I32, I8, U8, BOOL, UNKNOWN };

struct TensorInfo {
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;  // points into the mmap
    size_t nbytes = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path);
    ~SafeTensors();
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const TensorInfo& get(const std::string& name) const;

    // Copy a tensor out as f32 into dst (dst must hold numel() floats).
    // Handles F32/F16/BF16 source dtypes. This is the ONE copy we allow:
    // bf16-on-disk -> f32-resident happens once at load time.
    void load_as_f32(const std::string& name, float* dst) const;

    std::vector<std::string> names() const;
    size_t tensor_count() const { return tensors_.size(); }

private:
    void* map_ = nullptr;
    size_t map_size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::map<std::string, TensorInfo> tensors_;
};

#include "safetensors.h"
#include "bf16.h"
#include "json.h"

#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

// 0 == "unknown, don't size-check".
static size_t dtype_size(DType d) {
    switch (d) {
        case DType::F32: case DType::I32: return 4;
        case DType::F16: case DType::BF16: return 2;
        case DType::I64: return 8;
        case DType::I8: case DType::U8: case DType::BOOL: return 1;
        case DType::UNKNOWN: default: return 0;
    }
}

static DType parse_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64") return DType::I64;
    if (s == "I32") return DType::I32;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    if (s == "BOOL") return DType::BOOL;
    return DType::UNKNOWN;
}

SafeTensors::SafeTensors(const std::string& path) {
    const uint8_t* base = nullptr;

#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE)
        throw std::runtime_error("safetensors: cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(fh, &sz);
    map_size_ = static_cast<size_t>(sz.QuadPart);
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) { CloseHandle(fh); throw std::runtime_error("safetensors: CreateFileMapping failed"); }
    map_ = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!map_) { CloseHandle(mh); CloseHandle(fh); throw std::runtime_error("safetensors: MapViewOfFile failed"); }
    file_handle_ = fh;
    mapping_handle_ = mh;
    base = static_cast<const uint8_t*>(map_);
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("safetensors: cannot open " + path);
    struct stat st;
    if (fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("safetensors: fstat failed"); }
    map_size_ = static_cast<size_t>(st.st_size);
    map_ = ::mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) { ::close(fd_); throw std::runtime_error("safetensors: mmap failed"); }
    base = static_cast<const uint8_t*>(map_);
#endif

    if (map_size_ < 8) throw std::runtime_error("safetensors: file too small");
    uint64_t header_len;
    std::memcpy(&header_len, base, 8);
    if (header_len > map_size_ - 8) throw std::runtime_error("safetensors: bad header length");

    std::string header(reinterpret_cast<const char*>(base + 8), header_len);
    const uint8_t* blob = base + 8 + header_len;
    const uint64_t blob_size = map_size_ - 8 - header_len;

    // Every offset is validated against the mapping. This is not paranoia: a
    // truncated download (see docs/step-log.md — curl silently cut the weights
    // at 72% and the file "looked done") produces a header that promises data
    // the file does not contain. Without this check the first symptom is a
    // SIGSEGV deep inside load_as_f32, or worse, silent garbage weights.
    JsonValue root = parse_json(header);
    for (const auto& [name, info] : root.obj) {
        if (name == "__metadata__") continue;
        TensorInfo t;
        t.dtype = parse_dtype(info["dtype"].as_string());
        for (const auto& d : info["shape"].arr) t.shape.push_back(d.as_int());

        int64_t begin = info["data_offsets"].arr[0].as_int();
        int64_t end   = info["data_offsets"].arr[1].as_int();
        if (begin < 0 || end < begin || static_cast<uint64_t>(end) > blob_size)
            throw std::runtime_error(
                "safetensors: tensor '" + name + "' offsets [" + std::to_string(begin) + "," +
                std::to_string(end) + ") escape the " + std::to_string(blob_size) +
                "-byte blob — the file is TRUNCATED or corrupt");

        t.data = blob + begin;
        t.nbytes = static_cast<size_t>(end - begin);

        size_t esz = dtype_size(t.dtype);
        if (esz != 0 && t.nbytes != static_cast<size_t>(t.numel()) * esz)
            throw std::runtime_error("safetensors: tensor '" + name + "' byte span " +
                                     std::to_string(t.nbytes) + " != numel*" +
                                     std::to_string(esz));
        tensors_.emplace(name, std::move(t));
    }
}

SafeTensors::~SafeTensors() {
#if defined(_WIN32)
    if (map_) UnmapViewOfFile(map_);
    if (mapping_handle_) CloseHandle(static_cast<HANDLE>(mapping_handle_));
    if (file_handle_) CloseHandle(static_cast<HANDLE>(file_handle_));
#else
    if (map_ && map_ != MAP_FAILED) ::munmap(map_, map_size_);
    if (fd_ >= 0) ::close(fd_);
#endif
}

const TensorInfo& SafeTensors::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("safetensors: no tensor named '" + name + "'");
    return it->second;
}

void SafeTensors::load_as_f32(const std::string& name, float* dst) const {
    const TensorInfo& t = get(name);
    int64_t n = t.numel();
    switch (t.dtype) {
        case DType::F32:
            std::memcpy(dst, t.data, static_cast<size_t>(n) * sizeof(float));
            break;
        case DType::BF16: {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data);
            for (int64_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
            break;
        }
        case DType::F16: {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(t.data);
            for (int64_t i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
            break;
        }
        default:
            throw std::runtime_error("safetensors: load_as_f32 unsupported dtype for '" + name + "'");
    }
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const auto& [k, v] : tensors_) out.push_back(k);
    return out;
}

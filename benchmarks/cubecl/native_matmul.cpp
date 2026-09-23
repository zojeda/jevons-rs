// Isolated GGML/HIP Q4_K products with the same fixture as q4k_bench.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
template<typename T> static std::vector<T> read(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open fixture");
    auto size = input.tellg();
    if (size < 0 || size % sizeof(T)) throw std::runtime_error("Invalid fixture size");
    std::vector<T> result(size / sizeof(T));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result.data()), size)) throw std::runtime_error("Cannot read fixture");
    return result;
}
static json errors(const std::vector<float> & actual, const std::vector<float> & reference) {
    if (actual.size() != reference.size() || actual.empty()) throw std::runtime_error("Output shape mismatch");
    double squared=0, norm=0, maximum=0, max_reference=0;
    for (size_t i=0; i<actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) throw std::runtime_error("Nonfinite output");
        double difference = double(actual[i]) - reference[i];
        squared += difference*difference; norm += double(reference[i])*reference[i];
        maximum = std::max(maximum, std::abs(difference));
        max_reference = std::max(max_reference, std::abs(double(reference[i])));
    }
    return {{"relative_rmse",std::sqrt(squared/std::max(norm,1e-30))},
            {"max_abs_error",maximum},{"normalized_max_error",maximum/std::max(max_reference,1e-30)}};
}
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
int main(int argc, char ** argv) try {
    if (argc!=3) throw std::runtime_error("Usage: native_matmul FIXTURE_DIRECTORY ROUNDS");
    auto dir=std::filesystem::path(argv[1]);
    int rounds=std::stoi(argv[2]);
    if (rounds<1 || rounds>1000) throw std::runtime_error("Rounds must be in 1..1000");
    json manifest; std::ifstream(dir/"manifest.json") >> manifest;
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cuda_init(0),ggml_backend_free);
    if (!backend) throw std::runtime_error("Cannot initialize HIP");
    json cases=json::array();
    for (const auto & c:manifest.at("cases")) {
        int64_t m=c.at("m"), k=c.at("k"), n=c.at("n");
        int64_t experts=c.value("experts",int64_t(1)), top_k=c.value("top_k",int64_t(1));
        bool routed=c.contains("ids");
        auto packed=read<uint8_t>(dir/c.at("packed").get<std::string>());
        auto input=read<float>(dir/c.at("input").get<std::string>());
        auto expected=read<float>(dir/c.at("expected").get<std::string>());
        std::unique_ptr<ggml_context,decltype(&ggml_free)> wctx(ggml_init({ggml_tensor_overhead()*4,nullptr,true}),ggml_free);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead(),nullptr,true}),ggml_free);
        if (!wctx || !ctx) throw std::runtime_error("Cannot allocate graph context");
        auto w=ggml_new_tensor_3d(wctx.get(),GGML_TYPE_Q4_K,k,n,experts);
        auto x=routed ? ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,k,1,m) : ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,k,m);
        auto ids=routed ? ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,top_k,m) : nullptr;
        auto y=routed ? ggml_mul_mat_id(ctx.get(),w,x,ids) : ggml_mul_mat(ctx.get(),w,x);
        auto graph=ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph,y);
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> wb(ggml_backend_alloc_ctx_tensors(wctx.get(),backend.get()),ggml_backend_buffer_free);
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> gb(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
        if (!wb || !gb) throw std::runtime_error("Cannot allocate GPU buffers");
        ggml_backend_buffer_set_usage(wb.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (packed.size()!=ggml_nbytes(w) || input.size()*4!=ggml_nbytes(x) || expected.size()*4!=ggml_nbytes(y))
            throw std::runtime_error("Fixture shape mismatch");
        ggml_backend_tensor_set(w,packed.data(),0,packed.size());
        ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
        if (routed) {
            auto routes=read<int32_t>(dir/c.at("ids").get<std::string>());
            if(routes.size()*4!=ggml_nbytes(ids)) throw std::runtime_error("Routing shape mismatch");
            for(auto id:routes) if(id<0 || id>=experts) throw std::runtime_error("Invalid expert ID");
            ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*4);
        }
        ggml_backend_synchronize(backend.get());
        auto compute=[&] {
            if (ggml_backend_graph_compute_async(backend.get(),graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
            ggml_backend_synchronize(backend.get());
        };
        auto warmup=Clock::now(); for(int i=0;i<3;++i) compute();
        double warmup_ms=elapsed(warmup);
        std::vector<float> output(expected.size()); json samples=json::array();
        for (int i=0;i<rounds;++i) {
            auto start=Clock::now(); compute(); double time=elapsed(start);
            ggml_backend_tensor_get(y,output.data(),0,output.size()*4);
            auto error=errors(output,expected);
            if (error.at("relative_rmse").get<double>()>manifest.at("max_relative_rmse").get<double>() ||
                error.at("normalized_max_error").get<double>()>manifest.at("max_normalized_error").get<double>())
                throw std::runtime_error("Matmul tolerance failed: "+error.dump());
            samples.push_back({{"elapsed_ms",time},{"error",error}});
        }
        cases.push_back({{"name",c.at("name")},{"shape_mkn",{m,k,n}},
            {"experts",experts},{"top_k",top_k},
            {"packed_bytes",packed.size()},{"warmup_ms",warmup_ms},{"samples",samples}});
    }
    std::cout << json({{"backend","Pinned GGML/HIP Q4_K"},{"rounds",rounds},
        {"model_sha256",manifest.at("model_sha256")},{"warmup_per_case",3},{"cases",cases},
        {"scope",manifest.value("scope",std::string("Isolated quantized matrix products. Synchronized host time excludes transfers."))}}).dump(2)<<'\n';
    return 0;
} catch (const std::exception & error) { std::cerr<<error.what()<<'\n'; return 1; }

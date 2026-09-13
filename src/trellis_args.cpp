#include "trellis_args.h"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace trellis {

void print_usage(const char* argv0, bool server) {
    if (server) {
        fprintf(stderr,
            "usage: %s [--host H] [--port P] [--models DIR] [--gpu N] [generation defaults...]\n",
            argv0);
    } else {
        fprintf(stderr,
            "usage: %s <image.png> <out.glb> [options]\n"
            "   or: %s --image <image.png> --output <out.glb> [options]\n",
            argv0, argv0);
    }
    fprintf(stderr,
        "\n"
        "  -i, --image PATH        input image                  (image->3D)\n"
        "  -o, --output PATH       output .glb                  (default model.glb)\n"
        "      --copyright TEXT    glTF asset.copyright metadata\n"
        "  -m, --models DIR        GGUF model directory\n"
        "      --gpu N             GPU index, <0 = CPU          (default 0)\n"
        "  -s, --seed N            RNG seed                     (default 42)\n"
        "      --res 512|1024|1536 geometry resolution\n"
        "      --max-tokens N      HR token budget              (default 49152)\n"
        "      --bg-removal MODE   threshold | birefnet   (default: auto -- a pre-matted\n"
        "                          image keeps its alpha; otherwise BiRefNet when its model\n"
        "                          is present. The plain threshold matte cuts out specular\n"
        "                          highlights, which the flow then turns into holes.)\n"
        "      --birefnet          alias for --bg-removal birefnet\n"
        "      --no-texture        geometry only\n"
        "      --xatlas            xatlas UV unwrap (default)\n"
        "      --box-uv            voxel-native box projection (faster)\n"
        "      --band N            narrow-band DC remesh band width (default: auto —\n"
        "                          res/512, i.e. 1 @512 / 2 @1024, which suppresses the\n"
        "                          res-1024 outer-skin speckle; N forces that width)\n"
        "      --faces N           QEM face target before UV bake (default: 300K @1024 /\n"
        "                          150K @512; min 1000)\n"
        "      --decim GRID        legacy cluster-grid decimation (default: quadric\n"
        "                          simplify to 300K faces @1024 / 150K @512; 0 = none)\n"
        "      --atlas PX          UV atlas size (default 2048 @1024 / 1024 @512)\n"
        "      --tex-res N         texture PBR resolution 512/1024 (default: auto — drops\n"
        "                          a dense res-1024 decode to a clean res-512 PBR volume)\n"
        "      --webp on|off       encode GLB textures as WebP (default: on when built with\n"
        "                          WebP support; off = PNG)\n"
        "      --dump-bg           also write the background-removal cutout as <out>_cutout.png\n"
        "      --bg-only           background removal only: write the cutout and skip the rest\n"
        "      --f32               f32 sparse-conv compute\n"
        "      --no-fa             disable FlashAttention\n"
        "      --require-gpu       refuse CPU fallback\n"
        "      --gss F  --gsh F    guidance strengths\n"
        "      --steps N           flow sampler steps, all stages   (default 12)\n"
        "      --gi0 F  --gi1 F    guidance interval [gi0,gi1], all stages (default\n"
        "                          0.6 / 1.0, texture 0.9 — a step whose rescaled\n"
        "                          timestep falls outside runs one forward instead of\n"
        "                          the CFG pair. --steps 8 --gi0 0.75 is ~1.36x)\n"
        "      --host H  --port P  trellis-server bind address\n"
        "      --voxply            also dump the voxel point cloud as .ply\n"
        "      --dump-slat         dump the structured latent to disk\n"
        "  -h, --help              show this help\n");
}

namespace {

const char* skip_ws(const char* s) {
    while (*s && std::isspace((unsigned char)*s)) ++s;
    return s;
}

bool trailing_is_ws(const char* end) {
    return *skip_ws(end) == '\0';
}

}  // namespace

bool parse_int_strict(const char* s, int& out) {
    if (!s) return false;
    const char* b = skip_ws(s);
    if (!*b) return false;
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(b, &end, 10);
    if (end == b || !trailing_is_ws(end)) return false;
    if (errno == ERANGE || v < INT_MIN || v > INT_MAX) return false;
    out = (int)v;
    return true;
}

bool parse_float_strict(const char* s, float& out) {
    if (!s) return false;
    const char* b = skip_ws(s);
    if (!*b) return false;
    errno = 0;
    char* end = nullptr;
    const float v = std::strtof(b, &end);
    if (end == b || !trailing_is_ws(end)) return false;
    if (!std::isfinite(v)) return false;
    out = v;
    return true;
}

bool validate_sampling(const TrellisParams& p, std::string& err) {
    if (p.steps != -1 && p.steps <= 0) {
        err = "steps must be > 0 (got " + std::to_string(p.steps) + ")";
        return false;
    }
    auto bound = [&err](const char* name, float v) {
        if (v == -1.0f || (v >= 0.0f && v <= 1.0f)) return true;
        err = std::string(name) + " must be within [0,1] (got " + std::to_string(v) + ")";
        return false;
    };
    if (!bound("gi0", p.gi0) || !bound("gi1", p.gi1)) return false;
    // An unset bound still has to hold against the default it leaves in place,
    // so --gi1 0.5 alone is caught the same way --gi0 0.9 --gi1 0.5 is.
    const float lo = p.gi0 == -1.0f ? 0.6f : p.gi0;
    const float hi = p.gi1 == -1.0f ? 1.0f : p.gi1;
    if (lo > hi) {
        err = "gi0 (" + std::to_string(lo) + ") must be <= gi1 (" + std::to_string(hi) + ")";
        return false;
    }
    return true;
}

bool parse_args(int argc, char** argv, TrellisParams& p) {
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "[trellis] %s needs a value\n", name); return nullptr; }
            return argv[++i];
        };
        auto need = [&](const char* name) -> const char* {
            const char* v = next(name);
            return v;
        };

        if      (a == "-h" || a == "--help")    { p.help = true; return false; }
        else if (a == "-i" || a == "--image")   { const char* v = need(a.c_str()); if (!v) return false; p.image = v; }
        else if (a == "-o" || a == "--output")  { const char* v = need(a.c_str()); if (!v) return false; p.output = v; }
        else if (a == "--copyright")            { const char* v = need(a.c_str()); if (!v) return false; p.copyright = v; }
        else if (a == "-m" || a == "--models")  { const char* v = need(a.c_str()); if (!v) return false; p.models = v; }
        else if (a == "--gpu")                  { const char* v = need(a.c_str()); if (!v) return false; p.gpu = atoi(v); }
        else if (a == "-s" || a == "--seed")    { const char* v = need(a.c_str()); if (!v) return false; p.seed = (uint32_t)atoi(v); }
        else if (a == "--res")                  { const char* v = need(a.c_str()); if (!v) return false; p.set_res(atoi(v)); }
        else if (a == "--max-tokens")           { const char* v = need(a.c_str()); if (!v) return false; p.max_tokens = atoi(v); }
        else if (a == "--bg-removal")           { const char* v = need(a.c_str()); if (!v) return false; p.birefnet = (std::strcmp(v, "birefnet") == 0) ? 1 : 0; }
        else if (a == "--birefnet")             { p.birefnet = 1; }
        else if (a == "--no-texture")           { p.texture = false; }
        else if (a == "--xatlas")               { p.xatlas = true; }
        else if (a == "--box-uv")               { p.xatlas = false; }
        else if (a == "--band")                 { const char* v = need(a.c_str()); if (!v) return false; p.band = atoi(v); }
        else if (a == "--faces")                { const char* v = need(a.c_str()); if (!v) return false; p.faces = atoi(v); }
        else if (a == "--decim")                { const char* v = need(a.c_str()); if (!v) return false; p.decim = atoi(v); }
        else if (a == "--atlas" || a == "--tex"){ const char* v = need(a.c_str()); if (!v) return false; p.tex = atoi(v); }
        else if (a == "--tex-res")              { const char* v = need(a.c_str()); if (!v) return false; p.tex_res = atoi(v); }
        else if (a == "--webp")                 { const char* v = need(a.c_str()); if (!v) return false;
                                                  p.webp = (std::strcmp(v,"off")==0 || std::strcmp(v,"0")==0 || std::strcmp(v,"false")==0) ? 0
                                                         : (std::strcmp(v,"on")==0 || std::strcmp(v,"1")==0 || std::strcmp(v,"true")==0) ? 1 : -1; }
        else if (a == "--dump-bg")              { p.dump_bg = true; }
        else if (a == "--bg-only")              { p.bg_only = true; p.dump_bg = true; }
        else if (a == "--f32")                  { p.f32 = true; }
        else if (a == "--no-fa")                { p.no_fa = true; }
        else if (a == "--require-gpu")          { p.require_gpu = true; }
        else if (a == "--gss")                  { const char* v = need(a.c_str()); if (!v) return false; p.gss = (float)atof(v); }
        else if (a == "--gsh")                  { const char* v = need(a.c_str()); if (!v) return false; p.gsh = (float)atof(v); }
        else if (a == "--steps")                { const char* v = need(a.c_str()); if (!v) return false;
                                                  if (!parse_int_strict(v, p.steps)) { fprintf(stderr, "[trellis] --steps needs an integer, got: %s\n", v); return false; } }
        else if (a == "--gi0")                  { const char* v = need(a.c_str()); if (!v) return false;
                                                  if (!parse_float_strict(v, p.gi0)) { fprintf(stderr, "[trellis] --gi0 needs a number, got: %s\n", v); return false; } }
        else if (a == "--gi1")                  { const char* v = need(a.c_str()); if (!v) return false;
                                                  if (!parse_float_strict(v, p.gi1)) { fprintf(stderr, "[trellis] --gi1 needs a number, got: %s\n", v); return false; } }
        else if (a == "--host")                 { const char* v = need(a.c_str()); if (!v) return false; p.host = v; }
        else if (a == "--port")                 { const char* v = need(a.c_str()); if (!v) return false; p.port = atoi(v); }
        else if (a == "--voxply")               { p.voxply = true; }
        else if (a == "--dump-slat")            { p.dump_slat = true; }
        else if (!a.empty() && a[0] == '-')     { fprintf(stderr, "[trellis] unknown option: %s\n", a.c_str()); return false; }
        else if (positional == 0)               { p.image  = a; positional = 1; }
        else if (positional == 1)               { p.output = a; positional = 2; }
        else                                    { fprintf(stderr, "[trellis] unexpected argument: %s\n", a.c_str()); return false; }
    }
    std::string err;
    if (!validate_sampling(p, err)) {
        fprintf(stderr, "[trellis] %s\n", err.c_str());
        return false;
    }
    return true;
}

}  // namespace trellis

#pragma once
#include <cstdint>
#include <string>

namespace trellis {

// Cross-module runtime flags. Set from parsed params at the start of trellis_run;
// the modules that own them read them with an environment fallback so test
// binaries (which don't parse args) keep their historical TRELLIS_* behavior.
extern bool g_sparse_cast_f32;  // defined in sparse.cpp        (TRELLIS_F32)
extern bool g_no_fa;            // defined in dit.cpp           (TRELLIS_NOFA)
extern bool g_require_gpu;      // defined in trellis_model.cpp (TRELLIS_REQUIRE_GPU)

// Every knob for one TRELLIS.2 image->3D run. Resolved as default -> environment
// (the historical TRELLIS_* / GSS / GSH names) -> CLI flag, with the CLI winning.
// trellis-cli and trellis-server share the parser: the server runs it once for its
// launch defaults, then per request to apply overrides (resolution, bg removal, ...).
struct TrellisParams {
    std::string image;                                          // input image (image->3D)
    std::string output = "model.glb";                           // output .glb
    std::string copyright;                                      // glTF asset.copyright metadata
    std::string models = "models";              // GGUF dir; override with --models DIR
    std::string host   = "127.0.0.1";                           // trellis-server only
    int      port = 8080;                                       // trellis-server only
    int      gpu  = 0;                                          // >=0 GPU index, <0 CPU
    uint32_t seed = 0;

    bool cascade    = true;     // 1024 cascade (default); --res 512 selects the light path
    int  hr_res     = 1024;     // HR cascade target resolution (1024 / 1536)
    int  max_tokens = 49152;    // HR token budget (backoff floors at 1024)

    int birefnet = -1;          // bg removal: 1 BiRefNet, 0 white-threshold, -1 auto
                                // (auto: keep a pre-matted image's alpha; else BiRefNet when
                                // birefnet.gguf is present; threshold as last resort. The
                                // threshold matte reads specular highlights [min(RGB)>=232]
                                // as background and the model then generates holes there.)
    bool texture  = true;       // texture flow + UV bake (else geometry-only)
    bool xatlas   = true;       // xatlas UV unwrap (else voxel-native box projection)
    int  band     = 0;          // narrow-band DC remesh band width (remesh_dc.h).
                                //   0 = auto: scale with resolution (res/512) so the
                                //   smoothing offset is resolution-independent — 1 @512,
                                //   2 @1024 — which suppresses the res-1024 "outer-skin"
                                //   speckle (issue #22). >0 forces that width (e.g. 1 for
                                //   the thin-wall reference look, 2 for a thicker shell).
    int  faces    = -1;         // QEM face target before UV bake (-1 => 150K@512 / 300K@cascade)

    // Sampler overrides. -1 keeps each stage's tuned default (12 steps, and the
    // guidance interval each stage was tuned with) so an unset run is bit-identical
    // to one built before these existed.
    int  steps       = -1;      // every flow stage      (--steps)
    int  steps_ss    = -1;      // sparse-structure flow (--steps-ss),    overrides --steps
    int  steps_shape = -1;      // shape-SLAT flow       (--steps-shape), overrides --steps
    int  steps_tex   = -1;      // texture-SLAT flow     (--steps-tex),   overrides --steps
    // Guidance interval on the two GUIDED flows (sparse-structure, shape SLAT).
    // A step only pays for the second, unconditional forward while the (rescaled)
    // timestep is inside [gi0, gi1] — which is why a 12-step stage reports 22 and
    // not 24 forwards. Narrowing the interval buys wall clock without dropping
    // steps. The texture flow runs at guidance strength 1.0, i.e. unguided, so the
    // interval is inert there and these do not touch it.
    float gi0 = -1.0f;          // low end  (--gi0), stage default 0.6
    float gi1 = -1.0f;          // high end (--gi1), stage default 1.0
    int  decim    = -1;         // decimation cluster grid   (-1 => per-cascade default)
    int  tex      = -1;         // UV atlas size in px        (-1 => per-cascade default)
    int  tex_res  = -1;         // texture PBR resolution: -1 => auto (drop dense res-1024 tex to
                                //   512, whose clean coarse PBR bakes onto the res-1024 mesh
                                //   without the partial-coverage "skin" speckle); else force 512/1024
    int  webp     = -1;         // GLB texture encoding: -1 auto (WebP if built with it), 1 on, 0 off (PNG)
    bool f32      = false;      // f32 sparse-conv compute
    bool no_fa    = false;      // disable FlashAttention (manual softmax)
    bool require_gpu = false;   // refuse CPU fallback if no GPU is usable
    float gss = 7.5f;           // sparse-structure guidance strength
    float gsh = 7.5f;           // shape-SLAT guidance strength
    bool voxply = false;        // dump out/myvox.ply              (debug)
    bool dump_slat = false;     // dump /tmp/hr_slat.bin           (debug)
    bool dump_bg = false;       // also write the bg-removal cutout as <out>_cutout.png
    bool bg_only = false;       // background removal only: write the cutout and skip the rest

    bool help = false;          // --help requested

    // Per-stage step count: the stage flag wins, then --steps, then -1 = the
    // stage's own default (the caller substitutes it; the parser has no business
    // knowing that every stage happens to use 12 today).
    int steps_ss_or_default() const    { return steps_ss    > 0 ? steps_ss    : steps; }
    int steps_shape_or_default() const { return steps_shape > 0 ? steps_shape : steps; }
    int steps_tex_or_default() const   { return steps_tex   > 0 ? steps_tex   : steps; }

    // 512 -> light single-res path; 1024/1536 -> cascade with that HR target.
    void set_res(int res) {
        if (res <= 512) { cascade = false; hr_res = 512; }
        else            { cascade = true;  hr_res = res; }
    }
};

void print_usage(const char* argv0, bool server);

// Apply environment fallbacks, then parse argv (CLI wins). The first two bare
// (non-flag) positionals fill `image` then `output`. Returns false on a parse
// error OR when --help was requested; check p.help to tell them apart.
bool parse_args(int argc, char** argv, TrellisParams& p);

}  // namespace trellis

// Argument-parsing / sampling-validation checks for trellis-cli and trellis-server.
//   trellis-test-args          (no model, no GPU — pure parser, runs anywhere)
//
// The sampling flags (--steps/--gi0/--gi1) are the only knobs a caller can use
// to trade sampler forwards for geometry, so a silently-accepted bad value
// would cost a full pipeline run before anything noticed.
#include "trellis_args.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    printf("  %-58s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

// parse_args takes argv, so build one (argv[0] is the program name).
bool parse(const std::vector<std::string>& args, trellis::TrellisParams& p) {
    std::vector<char*> argv;
    std::vector<std::string> owned = args;
    std::string prog = "trellis-cli";
    argv.push_back(&prog[0]);
    for (auto& a : owned) argv.push_back(&a[0]);
    return trellis::parse_args((int)argv.size(), argv.data(), p);
}

bool accepts(const std::vector<std::string>& args) {
    trellis::TrellisParams p;
    return parse(args, p);
}

}  // namespace

int main() {
    printf("[strict numeric parsing]\n");
    {
        int i = -99; float f = -99.0f;
        check(trellis::parse_int_strict("8", i) && i == 8, "parse_int_strict(\"8\") -> 8");
        check(trellis::parse_int_strict(" 8 ", i) && i == 8, "parse_int_strict(\" 8 \") -> 8");
        check(!trellis::parse_int_strict("abc", i), "parse_int_strict(\"abc\") rejected");
        check(!trellis::parse_int_strict("8x", i), "parse_int_strict(\"8x\") rejected");
        check(!trellis::parse_int_strict("", i), "parse_int_strict(\"\") rejected");
        check(trellis::parse_float_strict("0.75", f) && f == 0.75f, "parse_float_strict(\"0.75\") -> 0.75");
        check(!trellis::parse_float_strict("abc", f), "parse_float_strict(\"abc\") rejected");
        check(!trellis::parse_float_strict("nan", f), "parse_float_strict(\"nan\") rejected");
    }

    printf("[defaults]\n");
    {
        trellis::TrellisParams p;
        check(parse({}, p), "no flags parses");
        check(p.steps == -1 && p.gi0 == -1.0f && p.gi1 == -1.0f,
              "unset sampling flags stay -1 (per-stage defaults)");
    }

    printf("[accepted sampling flags]\n");
    {
        trellis::TrellisParams p;
        check(parse({"--steps", "8", "--gi0", "0.75"}, p), "--steps 8 --gi0 0.75 parses");
        check(p.steps == 8 && p.gi0 == 0.75f && p.gi1 == -1.0f,
              "--steps 8 --gi0 0.75 leaves gi1 unset");
        check(accepts({"--steps", "1"}), "--steps 1 accepted");
        check(accepts({"--gi0", "0", "--gi1", "1"}), "--gi0 0 --gi1 1 accepted");
        check(accepts({"--gi0", "0.5", "--gi1", "0.5"}), "--gi0 0.5 --gi1 0.5 accepted (empty-but-valid)");
    }

    printf("[rejected sampling flags]\n");
    {
        check(!accepts({"--steps", "0"}),                  "--steps 0 rejected");
        check(!accepts({"--steps", "-3"}),                 "--steps -3 rejected");
        check(!accepts({"--steps", "abc"}),                "--steps abc rejected");
        check(!accepts({"--gi0", "1.5"}),                  "--gi0 1.5 rejected");
        check(!accepts({"--gi1", "-0.2"}),                 "--gi1 -0.2 rejected");
        check(!accepts({"--gi0", "0.9", "--gi1", "0.5"}),  "--gi0 0.9 --gi1 0.5 rejected");
        check(!accepts({"--gi1", "0.5"}),                  "--gi1 0.5 alone rejected (< default gi0 0.6)");
        check(!accepts({"--steps"}),                       "--steps with no value rejected");
    }

    printf("[validate_sampling on a server-side param set]\n");
    {
        trellis::TrellisParams p; std::string err;
        p.steps = 8; p.gi0 = 0.75f;
        check(trellis::validate_sampling(p, err), "steps 8 / gi0 0.75 validates");
        p.steps = 0;
        check(!trellis::validate_sampling(p, err) && !err.empty(), "steps 0 reports a reason");
    }

    printf(failures ? "\nFAILED (%d)\n" : "\nall checks passed\n", failures);
    return failures ? 1 : 0;
}

// 拼图快速求解器（C++）——边长配对预计算 + 深度优先装配
//
// 思路（用户提的"用配对信息"，也是经典拼图做法）：
//   1) 边长匹配只当**信息**用，不动任何缩放：接缝两侧边长差 ≤ max(rel*min, abs) 才算候选。
//      一条边可以与多条边匹配（A-B、A-C 都保留），这是信息，不是约束。
//   2) 相对位姿预计算：把「Q 的第 j 条边贴到 P 的第 i 条边」的相对变换算一次存起来
//      （在 P 的局部坐标系里），装配时只做坐标合成 → 快。
//   3) 深度优先 + 每层保留最好的 K 个状态（可按值剪枝），最后输出前 N 个完整解。
//
// 输入（纯文本，Python 侧生成，避免在这里解析 JSON）：
//   <N>
//   <id> <area> <nv>  x y  x y ...
// 输出：JSON 到 stdout
//   {"solutions":[{"gap_pct":..,"placed":[[id,rot,tx,ty],..],"polys":[[[x,y],..],..]}, ...]}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

struct Pt { double x, y; };
static inline Pt operator+(Pt a, Pt b) { return {a.x + b.x, a.y + b.y}; }
static inline Pt operator-(Pt a, Pt b) { return {a.x - b.x, a.y - b.y}; }
static inline Pt operator*(Pt a, double s) { return {a.x * s, a.y * s}; }
static inline double cross(Pt a, Pt b) { return a.x * b.y - a.y * b.x; }
static inline double norm2(Pt a) { return a.x * a.x + a.y * a.y; }
static inline double norm_(Pt a) { return std::sqrt(norm2(a)); }

struct Edge { Pt a, b; double len; };
struct Piece {
    int id;
    double area;
    std::vector<Pt> poly;
    std::vector<Edge> edges;
};

static Pt rot_p(Pt p, double th) {
    double c = std::cos(th), s = std::sin(th);
    return {c * p.x - s * p.y, s * p.x + c * p.y};
}

// 点在多边形内（射线法）
static bool point_in_poly(Pt p, const std::vector<Pt>& poly, double eps) {
    bool in = false;
    size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const Pt& a = poly[i];
        const Pt& b = poly[j];
        if (((a.y > p.y + eps) != (b.y > p.y + eps))) {
            double xin = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (p.x < xin) in = !in;
        }
    }
    return in;
}

// 近似重叠面积：在两者包围盒交集里按 step 采样；累计超过 limit 立刻返回（早退）
static double overlap_area_limited(const std::vector<Pt>& A, const std::vector<Pt>& B,
                                   double limit, double step = 1.6) {
    double ax0 = 1e18, ax1 = -1e18, ay0 = 1e18, ay1 = -1e18;
    for (const auto& p : A) { ax0 = std::min(ax0, p.x); ax1 = std::max(ax1, p.x);
                              ay0 = std::min(ay0, p.y); ay1 = std::max(ay1, p.y); }
    double bx0 = 1e18, bx1 = -1e18, by0 = 1e18, by1 = -1e18;
    for (const auto& p : B) { bx0 = std::min(bx0, p.x); bx1 = std::max(bx1, p.x);
                              by0 = std::min(by0, p.y); by1 = std::max(by1, p.y); }
    double x0 = std::max(ax0, bx0), x1 = std::min(ax1, bx1);
    double y0 = std::max(ay0, by0), y1 = std::min(ay1, by1);
    if (x1 <= x0 || y1 <= y0) return 0.0;              // 包围盒不相交
    double cell = step * step;
    double acc = 0.0;
    for (double y = y0; y <= y1; y += step) {
        for (double x = x0; x <= x1; x += step) {
            Pt p{x, y};
            if (point_in_poly(p, A, 0.0) && point_in_poly(p, B, 0.0)) {
                acc += cell;
                if (acc > limit) return acc;           // 够了，不用再算
            }
        }
    }
    return acc;
}


// 点到多边形边界的最短距离
static double dist_to_poly(Pt p, const std::vector<Pt>& poly) {
    double best = 1e18;
    size_t n = poly.size();
    for (size_t i = 0; i < n; ++i) {
        Pt a = poly[i], b = poly[(i + 1) % n];
        Pt ab = b - a, ap = p - a;
        double L2 = norm2(ab);
        double t = L2 > 1e-12 ? std::max(0.0, std::min(1.0, (ap.x * ab.x + ap.y * ab.y) / L2)) : 0.0;
        Pt proj = a + ab * t;
        double d = norm_(p - proj);
        if (d < best) best = d;
    }
    return best;
}

// 点是否在内部且「深入」超过 margin（margin 用于容忍 2~4px 的切割噪声）
static bool inside_deep(Pt p, const std::vector<Pt>& poly, double margin) {
    if (!point_in_poly(p, poly, 0.0)) return false;
    return dist_to_poly(p, poly) > margin;
}

// 明显重叠才算坏：顶点深入对方超过 margin 才判重叠
static bool overlap_bad(const std::vector<Pt>& A, const std::vector<Pt>& B, double margin) {
    for (const auto& p : B) if (inside_deep(p, A, margin)) return true;
    for (const auto& p : A) if (inside_deep(p, B, margin)) return true;
    return false;
}

// 凸包面积（Andrew monotone chain）
static double hull_area(const std::vector<Pt>& pts) {
    if (pts.size() < 3) return 0.0;
    std::vector<Pt> p = pts;
    std::sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    std::vector<Pt> h;
    h.reserve(p.size() * 2);
    for (size_t i = 0; i < p.size(); ++i) {
        while (h.size() >= 2 && cross(h[h.size() - 1] - h[h.size() - 2], p[i] - h[h.size() - 2]) <= 0)
            h.pop_back();
        h.push_back(p[i]);
    }
    size_t lower = h.size();
    for (int i = (int)p.size() - 2; i >= 0; --i) {
        while (h.size() > lower && cross(h[h.size() - 1] - h[h.size() - 2], p[i] - h[h.size() - 2]) <= 0)
            h.pop_back();
        h.push_back(p[i]);
    }
    if (h.size() < 3) return 0.0;
    double s = 0;
    for (size_t i = 0; i < h.size(); ++i) {
        const Pt& a = h[i];
        const Pt& b = h[(i + 1) % h.size()];
        s += cross(a, b);
    }
    return std::fabs(s) * 0.5;
}

struct Joint {  // 「把 Q 的第 j 条边贴到 P 的第 i 条边」的相对位姿（P 的局部坐标系）
    int q, j;
    double rot, tx, ty;
};

struct Placed { int id; double rot, tx, ty; };
struct State {
    std::vector<Placed> placed;
    std::vector<std::vector<Pt>> polys;   // 世界坐标
    std::vector<char> used_piece;
    std::vector<std::vector<char>> used_edge;   // [piece][edge]
    std::vector<std::pair<int, int>> pairs;     // 已用的 (P,i)-(Q,j)，用编码存
    double sum_area = 0;
    double score = 0;
    double gap = 0, rect = 0;
    std::string sig;
};

static void eval_state(State& st) {
    std::vector<Pt> pts;
    for (const auto& p : st.polys) pts.insert(pts.end(), p.begin(), p.end());
    double ha = hull_area(pts);
    double minx = 1e18, maxx = -1e18, miny = 1e18, maxy = -1e18;
    for (const auto& p : pts) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }
    double box = (maxx - minx) * (maxy - miny);
    st.gap = ha > 1e-9 ? (ha - st.sum_area) / ha * 100.0 : 999.0;
    st.rect = box > 1e-9 ? ha / box : 0.0;
    // 空隙为主；矩形度只做轻微 tie-break（不假设目标一定是方形）
    st.score = st.gap + 10.0 * (1.0 - st.rect);
}

static std::string make_sig(const State& st) {
    std::string s;
    char buf[96];
    for (const auto& p : st.placed) {
        int rotq = (int)std::lround(p.rot * 180.0 / M_PI / 4.0);
        std::snprintf(buf, sizeof(buf), "%d:%d:%d:%d|", p.id, rotq,
                      (int)std::lround(p.tx / 12.0), (int)std::lround(p.ty / 12.0));
        s += buf;
    }
    std::sort(s.begin(), s.end());
    return s;
}

int main(int argc, char** argv) {
    // 参数：--rel <相对容差> --abs <绝对容差> --beam <K> --tops <N> --nodes <上限>
    double rel_tol = 0.15, abs_tol = 9.0;
    double ovl_frac = 0.004, ovl_px = 60.0;      // 单对容许的重叠（面积占比 / 绝对像素²）
    int beamK = 3000, topN = 50;
    long nodes_cap = 4000000;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rel") && i + 1 < argc) rel_tol = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--abs") && i + 1 < argc) abs_tol = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--beam") && i + 1 < argc) beamK = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tops") && i + 1 < argc) topN = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--nodes") && i + 1 < argc) nodes_cap = atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--ovl-frac") && i + 1 < argc) ovl_frac = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--ovl-px") && i + 1 < argc) ovl_px = atof(argv[++i]);
    }

    // 读输入
    int n = 0;
    if (std::scanf("%d", &n) != 1 || n <= 0) { std::fprintf(stderr, "bad input\n"); return 2; }
    std::vector<Piece> pieces(n);
    for (int i = 0; i < n; ++i) {
        int id, nv;
        double area;
        if (std::scanf("%d %lf %d", &id, &area, &nv) != 3) return 2;
        Piece p;
        p.id = id;
        p.area = area;
        p.poly.resize(nv);
        for (int k = 0; k < nv; ++k) {
            double x, y;
            if (std::scanf("%lf %lf", &x, &y) != 2) return 2;
            p.poly[k] = {x, y};
        }
        p.edges.resize(nv);
        for (int k = 0; k < nv; ++k) {
            Pt a = p.poly[k], b = p.poly[(k + 1) % nv];
            p.edges[k] = {a, b, norm_(b - a)};
        }
        pieces[i] = p;
    }

    // ① 预计算相对位姿（边长匹配 → 每个候选位姿）
    std::vector<std::vector<std::vector<Joint>>> rel(n);
    long cand_pairs = 0;
    for (int P = 0; P < n; ++P) {
        rel[P].resize(pieces[P].edges.size());
        for (size_t i = 0; i < pieces[P].edges.size(); ++i) {
            const Edge& ei = pieces[P].edges[i];
            for (int Q = 0; Q < n; ++Q) {
                if (Q == P) continue;
                for (size_t j = 0; j < pieces[Q].edges.size(); ++j) {
                    const Edge& ej = pieces[Q].edges[j];
                    double d = std::fabs(ei.len - ej.len);
                    if (d > std::max(rel_tol * std::min(ei.len, ej.len), abs_tol)) continue;
                    // 两种贴法（正贴 / 反贴）
                    for (int anchor = 0; anchor < 2; ++anchor) {
                        Pt srcA = pieces[Q].poly[j];
                        Pt srcB = pieces[Q].poly[(j + 1) % pieces[Q].poly.size()];
                        Pt tgtA = anchor ? ei.b : ei.a;
                        Pt tgtB = anchor ? ei.a : ei.b;
                        Pt ds = srcB - srcA;
                        Pt dt = tgtB - tgtA;              // 反向：Q 的边朝 tgtA→tgtB
                        double ang = std::atan2(dt.y, dt.x) - std::atan2(ds.y, ds.x);
                        Pt t = tgtA - rot_p(srcA, ang);
                        // 校验：Q 的质心必须落在 P 的另一侧
                        Pt cen{0, 0};
                        for (const auto& p : pieces[Q].poly) cen = cen + p;
                        cen = cen * (1.0 / pieces[Q].poly.size());
                        Pt cenW = rot_p(cen, ang) + t;
                        Pt cenP{0, 0};
                        for (const auto& p : pieces[P].poly) cenP = cenP + p;
                        cenP = cenP * (1.0 / pieces[P].poly.size());
                        Pt u = (ei.b - ei.a) * (1.0 / std::max(ei.len, 1e-9));
                        double sa = cross(u, cenP - ei.a);
                        double sq = cross(u, cenW - ei.a);
                        if (sa * sq >= 0) continue;
                        // 校验：贴上去不能和 P 重叠
                        std::vector<Pt> qw;
                        qw.reserve(pieces[Q].poly.size());
                        for (const auto& p : pieces[Q].poly) qw.push_back(rot_p(p, ang) + t);
                        {
                            // 无条件按面积判重叠：细条状重叠顶点可能不深入，光靠顶点判会漏
                            double lim = std::max(ovl_frac * pieces[Q].area, ovl_px);
                            if (overlap_area_limited(pieces[P].poly, qw, lim) > lim) continue;
                        }
                        rel[P][i].push_back({Q, (int)j, ang, t.x, t.y});
                        ++cand_pairs;
                    }
                }
            }
        }
    }
    std::fprintf(stderr, "碎片 %d，候选贴合位姿 %ld\n", n, cand_pairs);

    // ② 深度优先装配（每层保留 beamK 个）
    int seed = 0;
    for (int i = 1; i < n; ++i) if (pieces[i].area > pieces[seed].area) seed = i;

    std::vector<State> layer;
    {
        State s;
        s.placed.push_back({pieces[seed].id, 0, 0, 0});
        s.polys.push_back(pieces[seed].poly);
        s.used_piece.assign(n, 0);
        s.used_piece[seed] = 1;
        s.used_edge.assign(n, {});
        for (int k = 0; k < n; ++k) s.used_edge[k].assign(pieces[k].edges.size(), 0);
        s.sum_area = pieces[seed].area;
        eval_state(s);
        s.sig = make_sig(s);
        layer.push_back(std::move(s));
    }

    std::vector<State> finished;
    long nodes = 0;

    for (int depth = 1; depth < n; ++depth) {
        std::vector<State> next;
        for (const auto& st : layer) {
            if (nodes > nodes_cap) break;
            // 遍历已放碎片的每条可用边
            for (size_t k = 0; k < st.placed.size(); ++k) {
                int P = -1;
                for (int t = 0; t < n; ++t) if (pieces[t].id == st.placed[k].id) { P = t; break; }
                if (P < 0) continue;
                for (size_t i = 0; i < pieces[P].edges.size(); ++i) {
                    if (st.used_edge[P][i]) continue;
                    for (const auto& jt : rel[P][i]) {
                        if (st.used_piece[jt.q]) continue;
                        if (st.used_edge[jt.q][jt.j]) continue;
                        // 合成世界位姿：Q = 该已放碎片的位姿 ∘ 相对位姿
                        const Placed& pp = st.placed[k];
                        double rot = pp.rot + jt.rot;
                        Pt t = rot_p({jt.tx, jt.ty}, pp.rot) + Pt{pp.tx, pp.ty};
                        std::vector<Pt> qw;
                        qw.reserve(pieces[jt.q].poly.size());
                        for (const auto& p : pieces[jt.q].poly) qw.push_back(rot_p(p, rot) + t);
                        // 与已放的全部多边形查重叠
                        bool bad = false;
                        double lim = std::max(ovl_frac * pieces[jt.q].area, ovl_px);
                        for (const auto& pw : st.polys) {
                            if (overlap_area_limited(pw, qw, lim) > lim) { bad = true; break; }
                        }
                        if (bad) continue;
                        State ns = st;
                        ns.placed.push_back({pieces[jt.q].id, rot, t.x, t.y});
                        ns.polys.push_back(qw);
                        ns.used_piece[jt.q] = 1;
                        ns.used_edge[jt.q][jt.j] = 1;
                        ns.used_edge[P][i] = 1;
                        ns.sum_area += pieces[jt.q].area;
                        eval_state(ns);
                        ns.sig = make_sig(ns);
                        next.push_back(std::move(ns));
                        ++nodes;
                    }
                }
            }
        }
        if (next.empty()) {
            std::fprintf(stderr, "第 %d 层无可行扩展\n", depth + 1);
            break;
        }
        // 排序 + 去重 + 截断
        std::sort(next.begin(), next.end(), [](const State& a, const State& b) { return a.score < b.score; });
        layer.clear();
        std::string last;
        for (auto& s : next) {
            if (!layer.empty() && s.sig == last) continue;
            last = s.sig;
            layer.push_back(std::move(s));
            if ((int)layer.size() >= beamK) break;
        }
        std::fprintf(stderr, "第 %d 层：候选 %zu，保留 %zu，最好 缝 %.2f%%（分 %.2f）\n",
                     depth + 1, next.size(), layer.size(), layer[0].gap, layer[0].score);
        if (depth == n - 1) {
            finished = layer;   // 最后一层就是完整解
        }
    }
    if (finished.empty() && (int)layer[0].placed.size() == n) finished = layer;

    std::sort(finished.begin(), finished.end(),
              [](const State& a, const State& b) { return a.score < b.score; });

    // ③ 输出 JSON（前 topN 个）
    std::printf("{\"n\":%d,\"solutions\":[", (int)finished.size());
    int emitted = 0;
    for (const auto& st : finished) {
        if (emitted >= topN) break;
        if (emitted) std::printf(",");
        std::printf("{\"gap_pct\":%.3f,\"rect\":%.4f,\"placed\":[", st.gap, st.rect);
        for (size_t i = 0; i < st.placed.size(); ++i) {
            const auto& p = st.placed[i];
            std::printf("%s[%d,%.6f,%.3f,%.3f]", i ? "," : "", p.id, p.rot, p.tx, p.ty);
        }
        std::printf("],\"polys\":[");
        for (size_t i = 0; i < st.polys.size(); ++i) {
            std::printf("%s[", i ? "," : "");
            for (size_t k = 0; k < st.polys[i].size(); ++k) {
                std::printf("%s[%.2f,%.2f]", k ? "," : "", st.polys[i][k].x, st.polys[i][k].y);
            }
            std::printf("]");
        }
        std::printf("]}");
        ++emitted;
    }
    std::printf("]}\n");
    std::fprintf(stderr, "输出 %d 个完整解（nodes=%ld）\n", emitted, nodes);
    return 0;
}

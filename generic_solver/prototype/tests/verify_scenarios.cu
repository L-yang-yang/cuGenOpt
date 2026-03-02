/**
 * verify_scenarios.cu - 验证所有测试场景的数据正确性
 * 
 * 纯CPU验证，不需要GPU。
 * 确保每个场景的最优解、评估函数、约束都正确。
 */

#include <cstdio>
#include <cmath>
#include "test_scenarios.h"

int main() {
    print_all_scenarios();
    
    int passed = 0, failed = 0;
    
    // ---- TSP5 ----
    {
        double d = TSP5::evaluate(TSP5::optimal_route, TSP5::N);
        printf("[TSP5] 最优路径距离 = %.1f (期望 %.1f) ", d, TSP5::optimal_distance);
        if (fabs(d - TSP5::optimal_distance) < 0.01) { printf("PASS\n"); passed++; }
        else { printf("FAIL\n"); failed++; }
        
        // 验证另一个解
        int route2[] = {0, 1, 3, 4, 2};
        double d2 = TSP5::evaluate(route2, TSP5::N);
        printf("[TSP5] 路径 0-1-3-4-2 距离 = %.1f\n", d2);
    }
    
    // ---- VRP4 ----
    {
        int route0[] = {1, 4};  // 路径0
        int route1[] = {2, 3};  // 路径1
        double dist0, load0, dist1, load1;
        VRP4::evaluate(route0, 2, dist0, load0);
        VRP4::evaluate(route1, 2, dist1, load1);
        
        double total_dist = dist0 + dist1;
        bool feasible = (load0 <= VRP4::capacity) && (load1 <= VRP4::capacity);
        
        printf("[VRP4] 路径0 [1,4]: 距离=%.1f 负载=%.1f (≤%.0f? %s)\n",
               dist0, load0, VRP4::capacity, load0 <= VRP4::capacity ? "是" : "否");
        printf("[VRP4] 路径1 [2,3]: 距离=%.1f 负载=%.1f (≤%.0f? %s)\n",
               dist1, load1, VRP4::capacity, load1 <= VRP4::capacity ? "是" : "否");
        printf("[VRP4] 总距离=%.1f 可行=%s ", total_dist, feasible ? "是" : "否");
        if (feasible) { printf("PASS\n"); passed++; }
        else { printf("FAIL\n"); failed++; }
        
        // 验证不可行解
        int bad0[] = {3, 4};
        double bd, bl;
        VRP4::evaluate(bad0, 2, bd, bl);
        printf("[VRP4] 不可行路径 [3,4]: 负载=%.1f (>%.0f? %s)\n",
               bl, VRP4::capacity, bl > VRP4::capacity ? "是,正确不可行" : "否,应该不可行!");
    }
    
    // ---- KNAPSACK6 ----
    {
        double w;
        double v = KNAPSACK6::evaluate(KNAPSACK6::optimal_selection, KNAPSACK6::N, w);
        bool feas = KNAPSACK6::is_feasible(KNAPSACK6::optimal_selection, KNAPSACK6::N);
        
        printf("[KNAPSACK6] 选{0,3,5}: 价值=%.1f 重量=%.1f 可行=%s ",
               v, w, feas ? "是" : "否");
        if (fabs(v - KNAPSACK6::optimal_value) < 0.01 && feas) { printf("PASS\n"); passed++; }
        else { printf("FAIL\n"); failed++; }
        
        // 验证不可行解
        int overweight[] = {1, 1, 1, 1, 0, 0};  // 物品0,1,2,3 重量=2+3+5+7=17>15
        bool feas2 = KNAPSACK6::is_feasible(overweight, KNAPSACK6::N);
        printf("[KNAPSACK6] 选{0,1,2,3}: 重量=17 可行=%s\n",
               feas2 ? "是,错误!" : "否,正确不可行");
    }
    
    // ---- ASSIGN4 ----
    {
        double c = ASSIGN4::evaluate(ASSIGN4::optimal_assign, ASSIGN4::N);
        printf("[ASSIGN4] 分配[1,2,0,3]: 成本=%.1f (期望%.1f) ",
               c, ASSIGN4::optimal_cost);
        if (fabs(c - ASSIGN4::optimal_cost) < 0.01) { printf("PASS\n"); passed++; }
        else { printf("FAIL\n"); failed++; }
        
        // 验证另一个解
        int assign2[] = {0, 1, 2, 3};  // 对角线分配
        double c2 = ASSIGN4::evaluate(assign2, ASSIGN4::N);
        printf("[ASSIGN4] 分配[0,1,2,3]: 成本=%.1f (对角线)\n", c2);
    }
    
    // ---- SCHEDULE3x4 ----
    {
        Solution sol;
        // 周一:员工1,3  周二:员工1,3  周三:员工0,2
        int sched[MAX_D1][MAX_D2] = {
            {0, 1, 0, 1},   // 周一: 员工1和3
            {0, 1, 0, 1},   // 周二: 员工1和3
            {1, 0, 1, 0}    // 周三: 员工0和2
        };
        memcpy(sol.data, sched, sizeof(sched));
        
        double tc, uf;
        bool feas;
        SCHEDULE3x4::evaluate(sol.data, tc, uf, feas);
        
        printf("[SCHEDULE] 排班方案: 成本=%.1f 不公平度=%.1f 可行=%s ",
               tc, uf, feas ? "是" : "否");
        if (feas && fabs(tc - 21.0) < 0.01) { printf("PASS\n"); passed++; }
        else { printf("FAIL\n"); failed++; }
        
        // 验证不可行解（周一安排3人）
        int bad_sched[MAX_D1][MAX_D2] = {
            {1, 1, 0, 1},   // 周一: 3人!
            {0, 1, 0, 1},
            {1, 0, 1, 0}
        };
        memcpy(sol.data, bad_sched, sizeof(bad_sched));
        SCHEDULE3x4::evaluate(sol.data, tc, uf, feas);
        printf("[SCHEDULE] 不可行排班(周一3人): 可行=%s\n",
               feas ? "是,错误!" : "否,正确不可行");
    }
    
    printf("\n========== 结果: %d PASS, %d FAIL ==========\n", passed, failed);
    return failed;
}

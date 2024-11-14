代码结构：
主程序：run_script.py，负责执行不同的策略模块。
策略模块：
strategy_risk_parity.py：固定权重策略和风险平价策略。
strategy_signal_based.py：基于信号的战术性调仓策略。
strategy_momentum.py：动量策略（您的最新需求）。
工具模块：
data_loader.py：数据加载和预处理模块，从Excel文件读取数据。
risk_budgeting.py：风险预算和风险平价算法的实现。
evaluator.py：策略绩效评估的实现。老的代码不能复用，把资产价格写死成变量，不能拓展
utils.py：其他公用的工具函数。

统一策略接口：
每个策略模块实现统一的run_strategy()方法，接受price_data、start_date、end_date和策略特定的parameters。
参数管理和策略配置：
在run_script.py中，定义策略配置列表strategies_config，每个策略有名称、模块、类和参数。
策略执行流程：
主程序循环遍历策略配置列表，动态导入策略模块和类，执行策略的run_strategy()方法。
将所有策略的评估结果存储在字典中，便于后续处理和比较。
结果评估和输出：
由Evaluator类统一计算各策略的绩效指标，包括累计收益、年化收益、波动率、夏普比率、最大回撤等。
在主程序中，输出各策略的绩效指标，方便比较和分析。
数据库上传（可选）：
如果需要，将各策略的结果上传到数据库中，以便在Metabase的dashboard中展示。
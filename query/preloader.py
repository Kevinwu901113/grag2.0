import time
import os
from typing import Dict, Any, Optional, Tuple
from llm.llm import LLMClient
from utils.io import load_chunks, load_vector_index, load_graph
from graph.graph_utils import extract_entity_names
from query.enhanced_retriever import EnhancedRetriever
from query.optimized_theme_matcher import ThemeMatcher
from query.reranker import SimpleReranker, LLMReranker
from query.context_scheduler import PriorityContextScheduler
from query.query_rewriter import QueryRewriter
from llm.answer_selector import AnswerSelector

class QueryPreloader:
    """
    查询预加载器
    
    在查询循环开始前预先加载所有必要的模型和组件，
    避免每次查询时重复加载，提升查询响应速度。
    """
    
    def __init__(self, config: Dict[str, Any], work_dir: str):
        """
        初始化预加载器
        
        Args:
            config: 配置字典
            work_dir: 工作目录
        """
        self.config = config
        self.work_dir = work_dir
        
        # 预加载的组件
        self.llm_client: Optional[LLMClient] = None
        self.chunks: Optional[list] = None
        self.vector_index: Optional[Tuple[Any, Dict[str, Any]]] = None
        self.graph: Optional[dict] = None
        self.entity_names: Optional[set] = None
        self.enhanced_retriever: Optional[EnhancedRetriever] = None
        self.theme_matcher: Optional[ThemeMatcher] = None
        self.simple_reranker: Optional[SimpleReranker] = None
        self.llm_reranker: Optional[LLMReranker] = None
        self.context_scheduler: Optional[PriorityContextScheduler] = None
        self.query_rewriter: Optional[QueryRewriter] = None
        self.answer_selector: Optional[AnswerSelector] = None
        
        # 预加载状态
        self.is_loaded = False
        self.load_time = 0.0
        
    def preload_all(self) -> None:
        """
        预加载所有必要的组件
        """
        if self.is_loaded:
            print("⚠️ 组件已经预加载，跳过重复加载")
            return
            
        print("\n🚀 开始预加载查询组件...")
        start_time = time.time()
        
        try:
            # 1. 预加载LLM客户端
            self._preload_llm_client()
            
            # 2. 预加载数据
            self._preload_data()
            
            # 3. 预加载检索器
            self._preload_retrievers()
            
            # 4. 预加载重排序器
            self._preload_rerankers()
            
            # 5. 预加载其他组件
            self._preload_other_components()
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            print(f"✅ 预加载完成，总耗时: {self.load_time:.2f}秒")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ 预加载失败: {e}")
            raise
    
    def _preload_llm_client(self) -> None:
        """
        预加载LLM客户端
        """
        print("📡 预加载LLM客户端...")
        start_time = time.time()
        
        self.llm_client = LLMClient(self.config)
        
        # 预热嵌入模型（发送一个测试文本）
        try:
            test_embeddings = self.llm_client.embed(["测试文本"], normalize_text=True, validate_dim=True)
            if test_embeddings:
                print(f"  ✅ 嵌入模型预热成功，维度: {len(test_embeddings[0])}")
            else:
                print("  ⚠️ 嵌入模型预热失败")
        except Exception as e:
            print(f"  ⚠️ 嵌入模型预热异常: {e}")
        
        elapsed = time.time() - start_time
        print(f"  ⏱️ LLM客户端加载耗时: {elapsed:.3f}秒")
    
    def _preload_data(self) -> None:
        """
        预加载数据文件
        """
        print("📚 预加载数据文件...")
        start_time = time.time()
        
        # 加载文档块
        self.chunks = load_chunks(self.work_dir)
        print(f"  ✅ 文档块加载完成，共 {len(self.chunks)} 个")
        
        # 加载向量索引
        try:
            self.vector_index = load_vector_index(self.work_dir)
            if self.vector_index[0] is not None:
                print(f"  ✅ 向量索引加载完成，索引大小: {self.vector_index[0].ntotal}")
            else:
                print("  ⚠️ 向量索引为空")
        except Exception as e:
            print(f"  ⚠️ 向量索引加载失败: {e}")
            self.vector_index = (None, {})
        
        # 加载图谱
        graph_path = os.path.join(self.work_dir, "graph.json")
        if os.path.exists(graph_path):
            try:
                self.graph = load_graph(self.work_dir)
                self.entity_names = extract_entity_names(self.graph) if self.graph else set()
                print(f"  ✅ 知识图谱加载完成，实体数量: {len(self.entity_names)}")
            except Exception as e:
                print(f"  ⚠️ 知识图谱加载失败: {e}")
                self.graph = None
                self.entity_names = set()
        else:
            print("  ℹ️ 知识图谱文件不存在，跳过加载")
            self.graph = None
            self.entity_names = set()
        
        elapsed = time.time() - start_time
        print(f"  ⏱️ 数据加载耗时: {elapsed:.3f}秒")
    
    def _preload_retrievers(self) -> None:
        """
        预加载检索器
        """
        print("🔍 预加载检索器...")
        start_time = time.time()
        
        # 预加载增强检索器
        try:
            self.enhanced_retriever = EnhancedRetriever(self.config, self.work_dir)
            print("  ✅ 增强检索器加载完成")
        except Exception as e:
            print(f"  ⚠️ 增强检索器加载失败: {e}")
            self.enhanced_retriever = None
        
        # 预加载主题匹配器
        try:
            self.theme_matcher = ThemeMatcher(self.chunks, self.config)
            print("  ✅ 主题匹配器加载完成")
        except Exception as e:
            print(f"  ⚠️ 主题匹配器加载失败: {e}")
            self.theme_matcher = None
        
        elapsed = time.time() - start_time
        print(f"  ⏱️ 检索器加载耗时: {elapsed:.3f}秒")
    
    def _preload_rerankers(self) -> None:
        """
        预加载重排序器
        """
        print("🔄 预加载重排序器...")
        start_time = time.time()
        
        rerank_config = self.config.get("rerank", {})
        
        # 预加载简单重排序器
        try:
            self.simple_reranker = SimpleReranker(rerank_config)
            print("  ✅ 简单重排序器加载完成")
        except Exception as e:
            print(f"  ⚠️ 简单重排序器加载失败: {e}")
            self.simple_reranker = None
        
        # 预加载LLM重排序器
        if self.llm_client:
            try:
                self.llm_reranker = LLMReranker(self.llm_client, rerank_config)
                print("  ✅ LLM重排序器加载完成")
            except Exception as e:
                print(f"  ⚠️ LLM重排序器加载失败: {e}")
                self.llm_reranker = None
        
        elapsed = time.time() - start_time
        print(f"  ⏱️ 重排序器加载耗时: {elapsed:.3f}秒")
    
    def _preload_other_components(self) -> None:
        """
        预加载其他组件
        """
        print("🔧 预加载其他组件...")
        start_time = time.time()
        
        # 预加载上下文调度器
        try:
            self.context_scheduler = PriorityContextScheduler(self.config)
            print("  ✅ 上下文调度器加载完成")
        except Exception as e:
            print(f"  ⚠️ 上下文调度器加载失败: {e}")
            self.context_scheduler = None
        
        # 预加载查询改写器
        try:
            from query.query_rewriter import is_query_rewrite_enabled
            if is_query_rewrite_enabled(self.config):
                self.query_rewriter = QueryRewriter(self.config)
                print("  ✅ 查询改写器加载完成")
            else:
                print("  ℹ️ 查询改写功能未启用，跳过加载")
        except Exception as e:
            print(f"  ⚠️ 查询改写器加载失败: {e}")
            self.query_rewriter = None
        
        # 预加载答案选择器
        if self.llm_client:
            try:
                answer_selector_config = self.config.get('answer_selector', {})
                self.answer_selector = AnswerSelector(self.llm_client, answer_selector_config)
                print("  ✅ 答案选择器加载完成")
            except Exception as e:
                print(f"  ⚠️ 答案选择器加载失败: {e}")
                self.answer_selector = None
        
        elapsed = time.time() - start_time
        print(f"  ⏱️ 其他组件加载耗时: {elapsed:.3f}秒")
    
    def get_llm_client(self) -> LLMClient:
        """
        获取预加载的LLM客户端
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.llm_client
    
    def get_chunks(self) -> list:
        """
        获取预加载的文档块
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.chunks
    
    def get_vector_index(self) -> Tuple[Any, Dict[str, Any]]:
        """
        获取预加载的向量索引
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.vector_index
    
    def get_graph_data(self) -> Tuple[Optional[dict], Optional[set]]:
        """
        获取预加载的图谱数据
        
        Returns:
            (graph, entity_names) 元组
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.graph, self.entity_names
    
    def get_enhanced_retriever(self) -> Optional[EnhancedRetriever]:
        """
        获取预加载的增强检索器
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.enhanced_retriever
    
    def get_theme_matcher(self) -> Optional[ThemeMatcher]:
        """
        获取预加载的主题匹配器
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.theme_matcher
    
    def get_reranker(self, reranker_type: str) -> Optional[Any]:
        """
        获取指定类型的重排序器
        
        Args:
            reranker_type: 重排序器类型 ('simple' 或 'llm')
        
        Returns:
            对应的重排序器实例
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        
        if reranker_type == "simple":
            return self.simple_reranker
        elif reranker_type == "llm":
            return self.llm_reranker
        else:
            return None
    
    def get_context_scheduler(self) -> Optional[PriorityContextScheduler]:
        """
        获取预加载的上下文调度器
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.context_scheduler
    
    def get_query_rewriter(self) -> Optional[QueryRewriter]:
        """
        获取预加载的查询改写器
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.query_rewriter
    
    def get_answer_selector(self) -> Optional[AnswerSelector]:
        """
        获取预加载的答案选择器
        """
        if not self.is_loaded:
            raise RuntimeError("组件尚未预加载，请先调用 preload_all()")
        return self.answer_selector
    
    def get_load_time(self) -> float:
        """
        获取预加载耗时
        """
        return self.load_time
    
    def is_preloaded(self) -> bool:
        """
        检查是否已预加载
        """
        return self.is_loaded
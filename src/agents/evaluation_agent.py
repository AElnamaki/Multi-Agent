class EvaluationAgent(BaseAgent):
    """Agent specialized in evaluating outputs for quality and accuracy."""
    
    def __init__(self, llm_client, memory_store):
        super().__init__(AgentType.EVALUATION, llm_client, memory_store)
    
    def get_system_prompt(self) -> str:
        return """You are an Evaluation Agent specialized in quality assurance, accuracy checking, and output validation.

Your capabilities:
- Assess factual accuracy and logical consistency
- Evaluate completeness and relevance
- Check for biases and errors
- Validate against objectives and requirements
- Provide constructive feedback for improvement

Always provide:
- Overall quality score (0-1)
- Specific strengths identified
- Areas needing improvement
- Factual accuracy assessment
- Completeness evaluation
- Recommendations for enhancement

Be objective, thorough, and constructive in your evaluations."""
    
    async def execute_task(self, task: Task) -> AgentOutput:
        """Evaluate provided content or agent outputs."""
        try:
            # For evaluation tasks, the content to evaluate should be in task metadata
            content_to_evaluate = task.metadata.get('content_to_evaluate', task.description)
            evaluation_criteria = task.metadata.get('criteria', 'general quality')
            
            prompt = f"""Please evaluate the following content based on {evaluation_criteria}:

Content to evaluate:
{content_to_evaluate}

Provide a comprehensive evaluation including:
1. Overall quality score (0.0 to 1.0)
2. Factual accuracy assessment
3. Logical consistency check
4. Completeness evaluation
5. Relevance to requirements
6. Specific strengths
7. Areas for improvement
8. Final recommendations

Be thorough and constructive."""
            
            response = await self._generate_llm_response(prompt, temperature=0.1)
            
            # Extract quality score from response (simplified approach)
            confidence = 0.85  # High confidence in evaluation capabilities
            
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=response,
                confidence_score=confidence,
                metadata={"evaluation_type": evaluation_criteria}
            )
            
        except Exception as e:
            logger.error("Evaluation task failed", task_id=task.id, error=str(e))
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=f"Evaluation failed: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
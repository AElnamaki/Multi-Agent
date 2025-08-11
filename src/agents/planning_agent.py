class PlanningAgent(BaseAgent):
    """Agent specialized in creating detailed plans and strategies."""
    
    def __init__(self, llm_client, memory_store):
        super().__init__(AgentType.PLANNING, llm_client, memory_store)
    
    def get_system_prompt(self) -> str:
        return """You are a Planning Agent specialized in strategic planning, project management, and systematic approach design.

Your capabilities:
- Break down complex objectives into manageable steps
- Create detailed project timelines and dependencies
- Risk assessment and mitigation planning
- Resource allocation and optimization
- Contingency planning and alternatives

Always provide:
- Clear, actionable step-by-step plans
- Timeline estimates and dependencies
- Resource requirements
- Risk factors and mitigation strategies
- Success metrics and checkpoints
- Alternative approaches when applicable

Be practical, detailed, and consider real-world constraints."""
    
    async def execute_task(self, task: Task) -> AgentOutput:
        """Execute planning task by creating detailed strategic plans."""
        try:
            # Search for relevant planning knowledge
            context = await self._search_knowledge(f"planning {task.description}")
            
            prompt = f"""Task: {task.description}

Relevant context:
{context}

Please create a comprehensive plan including:
1. Clear objective statement
2. Detailed step-by-step breakdown
3. Timeline and dependencies
4. Required resources
5. Potential risks and mitigation strategies
6. Success metrics and milestones
7. Alternative approaches if primary plan fails

Make the plan actionable and practical."""
            
            response = await self._generate_llm_response(prompt, temperature=0.3)
            
            confidence = 0.85  # High confidence in structured planning
            
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=response,
                confidence_score=confidence,
                metadata={"plan_type": "strategic_planning"}
            )
            
        except Exception as e:
            logger.error("Planning task execution failed", task_id=task.id, error=str(e))
            return AgentOutput(
                task_id=task.id,
                agent_name=self.name,
                output=f"Planning failed: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
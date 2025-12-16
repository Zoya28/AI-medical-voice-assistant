from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import operator
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


# ==========================================================
# State Definitions
# ==========================================================
class ConversationState(TypedDict):
    user_query: str
    intent: str
    intent_confidence: float
    entities: Dict[str, Any]
    last_intent: Optional[str]
    last_entities: Dict[str, Any]
    last_date: Optional[Any]
    last_metric: Optional[str]
    active_slots: Dict[str, Any]
    missing_slots: List[str]
    awaiting_clarification: bool
    clarification_question: Optional[str]
    retrieved_data: Optional[Dict[str, Any]]
    response: str
    messages: Annotated[List, operator.add]
    turn_count: int
    needs_entities: bool
    needs_data: bool
    needs_response: bool
    is_complete: bool
    session_id: str
    timestamp: str


class NodeDecision(Enum):
    CONTINUE = "continue"
    CLARIFY = "clarify"
    QUERY_DATA = "query_data"
    GENERATE_RESPONSE = "generate_response"
    END = "end"


# ==========================================================
# LangGraph Conversation Manager
# ==========================================================

class LangGraphConversationManager:
    """LangGraph Conversation Manager with consistent output."""

    __slots__ = (
        "max_history", "inference_window", "session_timeout",
        "memory", "graph", "app"
    )

    def __init__(self, max_history: int = 10, inference_window: int = 4, session_timeout: int = 600):
        self.max_history = max_history
        self.inference_window = inference_window
        self.session_timeout = session_timeout
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        print("LangGraph Conversation Manager initialized")

    FOLLOWUP_PREFIXES = ("what about", "how about", "and ", "also", "too", "as well", "what was", "how was")

    def _trim_messages(self, messages: List) -> None:
        """Trim messages list in-place to at most (max_history * 2) items."""
        limit = self.max_history * 2
        if len(messages) > limit:
            del messages[0: len(messages) - limit]

    # ==========================================================
    # Graph Construction
    # ==========================================================
    def _build_graph(self) -> StateGraph:
        wf = StateGraph(ConversationState)
        add_node = wf.add_node
        
        add_node("initialize", self._initialize_node)
        add_node("resolve_entities", self._resolve_entities_node)
        add_node("check_slots", self._check_slots_with_context_node)
        add_node("query_data", self._query_data_node)
        add_node("generate_response", self._generate_response_node)
        add_node("clarify", self._clarify_node)

        wf.set_entry_point("initialize")
        wf.add_conditional_edges("initialize", self._route_after_init, {"resolve_entities": "resolve_entities", "end": END})
        wf.add_conditional_edges("resolve_entities", self._route_after_entities, {"check_slots": "check_slots", "clarify": "clarify", "end": END})
        wf.add_conditional_edges("check_slots", self._route_after_slots, {"query_data": "query_data", "clarify": "clarify", "resolve_entities": "resolve_entities"})
        wf.add_edge("query_data", "generate_response")
        wf.add_edge("generate_response", END)
        wf.add_edge("clarify", END)
        return wf

    # ==========================================================
    # Node Implementations
    # ==========================================================
    def _initialize_node(self, state: ConversationState) -> ConversationState:
        """Initialize turn with proper context preservation."""
        state["turn_count"] = state.get("turn_count", 0) + 1
        state["timestamp"] = datetime.now().isoformat()
        state.update({
            "needs_entities": True,
            "needs_data": True,
            "needs_response": True,
            "is_complete": False,
            "awaiting_clarification": False
        })
        if (q := state.get("user_query")):
            messages = state.setdefault("messages", [])
            messages.append(HumanMessage(content=q))
            self._trim_messages(messages)
        return state

    def _resolve_entities_node(self, state: ConversationState) -> ConversationState:
        """Context-aware entity resolution with proper inheritance."""
        entities = state.get("entities") or {}
        intent = state.get("intent", "")
        query = (state.get("user_query") or "").lower()

        # Get context from previous turn
        last_date = state.get("last_date")
        last_metric = state.get("last_metric")
        last_intent = state.get("last_intent")

        is_followup = any(query.startswith(p) for p in self.FOLLOWUP_PREFIXES)
        current_intent = intent
        is_cross_intent = bool(last_intent and "query" in current_intent and "query" in last_intent and current_intent != last_intent)

        # Process dates with proper context
        dates = entities.get("dates")
        if (is_followup or is_cross_intent):
            # Check if we should inherit the date
            if not dates and last_date:
                # Inherit from previous turn
                entities["dates"] = [last_date]
                entities["_inferred_date"] = True
                entities["_cross_intent"] = is_cross_intent
            elif dates and len(dates) == 1:
                # Check if current date is "today" - don't override with old date
                current_date_text = str((dates[0] or {}).get("raw_text", "")).lower()
                if current_date_text not in ("today", "now"):
                    # Keep the current date
                    pass
                # If it IS today, keep it (don't inherit)

        # Process metrics with proper context
        if not entities.get("metrics"):
            # First try to infer from intent
            inferred = self._infer_metric_from_intent(intent)
            if inferred:
                entities["metrics"] = [{"type": inferred}]
                entities["_inferred_metric"] = True
            # Then try to inherit from previous turn
            elif (is_followup or is_cross_intent) and last_metric:
                entities["metrics"] = [last_metric]
                entities["_inferred_metric"] = True

        state["entities"] = entities
        state["needs_entities"] = False
        return state

    def _check_slots_with_context_node(self, state: ConversationState) -> ConversationState:
        """FIXED: Enhanced context-aware slot validation without duplication."""
        entities = state.get("entities", {})
        intent = state.get("intent", "")
        query = state.get("user_query", "").lower()
        required = self._get_required_slots(intent)

        # Fast path: no required slots
        if not required:
            state.update({"active_slots": {}, "missing_slots": []})
            return state

        # Context awareness checks
        is_followup = any(query.startswith(p) for p in self.FOLLOWUP_PREFIXES)
        contains_that_day = "that day" in query or "the same day" in query
        referring_to_previous = "it" in query or "this" in query or "that" in query

        missing = []
        active = {}
        dates = entities.get("dates")
        metrics = entities.get("metrics")
        last_date = state.get("last_date")
        last_metric = state.get("last_metric")
        
        # Process each required slot ONCE with context
        for slot, (stype, required_flag) in required.items():
            if stype == "date":
                if dates:
                    # Use current date from entities
                    active[slot] = {"filled": True, "value": dates[0]}
                elif (is_followup or contains_that_day or referring_to_previous) and last_date:
                    # Inherit from context
                    active[slot] = {"filled": True, "value": last_date, "inherited": True}
                else:
                    # No date available
                    active[slot] = {"filled": False, "value": None}
                    if required_flag:
                        missing.append(slot)
            
            elif stype == "metric":
                if metrics:
                    # Use current metric from entities
                    active[slot] = {"filled": True, "value": metrics[0]}
                elif (is_followup or referring_to_previous) and last_metric:
                    # Inherit from context
                    active[slot] = {"filled": True, "value": last_metric, "inherited": True}
                else:
                    # No metric available
                    active[slot] = {"filled": False, "value": None}
                    if required_flag:
                        missing.append(slot)

        state.update({"active_slots": active, "missing_slots": missing})
        return state

    def _query_data_node(self, state: ConversationState) -> ConversationState:
        """Mark that data retrieval is needed."""
        state["needs_data"] = False
        # Data will be retrieved externally and updated via update_with_data
        return state

    def _generate_response_node(self, state: ConversationState) -> ConversationState:
        """Finalize response and update context."""
        state.update({"needs_response": False, "is_complete": True})
        
        # Add response to messages if available
        if (resp := state.get("response")):
            messages = state.setdefault("messages", [])
            messages.append(AIMessage(content=resp))
            self._trim_messages(messages)

        # Update context for next turn - ONLY update if we have NEW values
        entities = state.get("entities") or {}
        
        # Update last_intent
        if state.get("intent"):
            state["last_intent"] = state.get("intent")
        
        # Update last_date ONLY if we have a date in current entities (not inherited)
        if (dates := entities.get("dates")) and not entities.get("_inferred_date"):
            state["last_date"] = dates[0]
        
        # Update last_metric ONLY if we have a metric in current entities (not inherited)
        if (metrics := entities.get("metrics")) and not entities.get("_inferred_metric"):
            state["last_metric"] = metrics[0]
        
        # Store the entities for reference
        state["last_entities"] = entities
        
        return state

    def _clarify_node(self, state: ConversationState) -> ConversationState:
        """Generate clarification question."""
        missing = state.get("missing_slots", [])
        if not missing:
            return state
        
        clarification = self._generate_clarification(state.get("intent", ""), missing[0])
        state.update({
            "clarification_question": clarification,
            "response": clarification,
            "awaiting_clarification": True,
            "is_complete": True
        })
        
        messages = state.setdefault("messages", [])
        messages.append(AIMessage(content=clarification))
        self._trim_messages(messages)
        return state

    # ==========================================================
    # Routing
    # ==========================================================
    def _route_after_init(self, state: ConversationState) -> str:
        return "resolve_entities" if state.get("user_query") else "end"

    def _route_after_entities(self, state: ConversationState) -> str:
        return "check_slots"

    def _route_after_slots(self, state: ConversationState) -> str:
        return "clarify" if state.get("missing_slots") else "query_data"

    # ==========================================================
    # Helpers
    # ==========================================================
    _intent_metric_map = {
        "query_steps": "steps",
        "query_calories": "calories",
        "query_sleep": "sleep",
        "query_heart_rate": "heart_rate",
        "query_distance": "distance"
    }

    _slot_templates = {
        "query_steps": {"date": ("date", False)},
        "query_calories": {"date": ("date", False)},
        "query_sleep": {"date": ("date", False)},
        "query_heart_rate": {"date": ("date", False)},
        "set_goal": {"goal_type": ("metric", True), "goal_value": ("number", True)},
        "compare_data": {"metric": ("metric", True), "date_range": ("date_range", True)}
    }

    _clarification_templates = {
        "date": "For which time period? (today, yesterday, last week, etc.)",
        "metric": "Which metric would you like to know about? (steps, calories, sleep, heart rate, etc.)",
        "goal_type": "What type of goal would you like to set? (steps, calories, sleep, etc.)",
        "goal_value": "What is your target value?",
        "date_range": "Which time periods would you like to compare?"
    }

    def _infer_metric_from_intent(self, intent: str) -> Optional[str]:
        return self._intent_metric_map.get(intent)

    def _get_required_slots(self, intent: str) -> Dict[str, tuple]:
        return self._slot_templates.get(intent, {})

    def _generate_clarification(self, intent: str, slot: str) -> str:
        return self._clarification_templates.get(slot, "Could you provide more details?")

    # ==========================================================
    # Public Interface
    # ==========================================================
    def process_turn(self, user_query: str, intent: str, intent_confidence: float, entities: Dict[str, Any], session_id: str = "default") -> Dict[str, Any]:
        """Process a conversation turn with proper state management."""
        config = {"configurable": {"thread_id": session_id}}
        
        # Get current state to preserve context
        current_state = self.get_state(session_id)
        
        input_state = {
            "user_query": user_query,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "entities": entities,
            "session_id": session_id,
            "messages": current_state.get("messages", []) if current_state else [],
            "turn_count": current_state.get("turn_count", 0) if current_state else 0,
            # Preserve context from previous turn
            "last_intent": current_state.get("last_intent") if current_state else None,
            "last_date": current_state.get("last_date") if current_state else None,
            "last_metric": current_state.get("last_metric") if current_state else None,
            "last_entities": current_state.get("last_entities", {}) if current_state else {},
        }
        
        return self.app.invoke(input_state, config)

    def get_state(self, session_id: str = "default") -> Optional[ConversationState]:
        """Get current conversation state."""
        config = {"configurable": {"thread_id": session_id}}
        try:
            snapshot = self.app.get_state(config)
            return snapshot.values if snapshot else None
        except Exception:
            return None

    def update_with_data(self, session_id: str, retrieved_data: Dict[str, Any]) -> Optional[ConversationState]:
        """Update state with retrieved data."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.app.get_state(config)
        if not snapshot:
            return None
        values = snapshot.values
        values["retrieved_data"] = retrieved_data
        self.app.update_state(config, values)
        return values

    def update_with_response(self, session_id: str, response: str) -> Optional[ConversationState]:
        """Update state with generated response."""
        config = {"configurable": {"thread_id": session_id}}
        snapshot = self.app.get_state(config)
        if not snapshot:
            return None
        values = snapshot.values
        values["response"] = response
        self.app.update_state(config, values)
        return values

    def get_conversation_history(self, session_id: str = "default", num_turns: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        state = self.get_state(session_id)
        if not state or not (msgs := state.get("messages")):
            return []
        msgs = msgs[-num_turns * 2:]
        return [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in msgs if isinstance(m, (HumanMessage, AIMessage))]

    def export_graph(self, filepath: str = "conversation_graph.png"):
        """Export conversation graph visualization."""
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(self.app.get_graph().draw_mermaid_png()))
            img.save(filepath)
            print(f"✅ Graph exported → {filepath}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
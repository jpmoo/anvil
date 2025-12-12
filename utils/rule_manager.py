"""Rule management for retrieval-based instruction conditioning"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from utils.config import get_model_data_dir, get_model_context_dir, get_model_behavioral_path
from utils.training_data import TrainingDataManager
from utils.ollama_client import OllamaClient


class RuleManager:
    """Manages rules generated from context, learning sessions, and behavior rules"""
    
    def __init__(self, model_name: str, base_model: str = "llama2"):
        self.model_name = model_name
        self.base_model = base_model
        self.data_manager = TrainingDataManager(model_name)
        self.rules_dir = get_model_data_dir(model_name)
        self.rules_file = self.rules_dir / "rules.json"
        self.ollama_client = OllamaClient(base_model)
    
    def generate_context_summaries(self) -> List[Dict]:
        """Generate summary rules from context files"""
        context_files = self.data_manager.get_all_context_files()
        rules = []
        
        for context_item in context_files:
            text = context_item["text"]
            metadata = context_item["metadata"]
            
            # Generate a concise summary/rule from the context
            summary_prompt = f"""Based on the following context, create a concise rule or guideline that captures the key information. The rule should be actionable and specific.

Context:
{text[:2000]}  # Limit to first 2000 chars for summary generation

Create a brief rule (1-3 sentences) that summarizes the key information from this context:"""
            
            try:
                # Use Ollama to generate summary
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that creates concise, actionable rules from context."},
                    {"role": "user", "content": summary_prompt}
                ]
                summary = self.ollama_client.chat(messages, stream=False)
                
                rules.append({
                    "type": "context",
                    "source": metadata.get("filename", "unknown"),
                    "rule": summary.strip(),
                    "metadata": metadata,
                    "date": datetime.now().isoformat()
                })
            except Exception as e:
                # Fallback: use first few sentences as rule
                sentences = text.split('.')[:3]
                fallback_rule = '. '.join(s for s in sentences if s.strip()) + '.'
                rules.append({
                    "type": "context",
                    "source": metadata.get("filename", "unknown"),
                    "rule": fallback_rule,
                    "metadata": metadata,
                    "date": datetime.now().isoformat(),
                    "fallback": True
                })
        
        return rules
    
    def generate_learning_session_summaries(self) -> List[Dict]:
        """Generate summary rules from learning session Q&A pairs"""
        learned_pairs = self.data_manager.get_learned_pairs()
        rules = []
        
        # Group pairs by topic/theme and create summaries
        if not learned_pairs:
            return rules
        
        # For each Q&A pair, create a rule
        for pair in learned_pairs:
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            
            if not question or not answer:
                continue
            
            # Generate a rule from the Q&A
            rule_prompt = f"""Based on this question and answer, create a concise rule or guideline:

Question: {question}
Answer: {answer}

Create a brief rule (1-2 sentences) that captures the key principle or information:"""
            
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that creates concise rules from Q&A pairs."},
                    {"role": "user", "content": rule_prompt}
                ]
                rule_text = self.ollama_client.chat(messages, stream=False)
                
                rules.append({
                    "type": "learning_session",
                    "source": "Q&A pair",
                    "rule": rule_text.strip(),
                    "question": question,
                    "answer": answer,
                    "date": pair.get("date", datetime.now().isoformat())
                })
            except Exception as e:
                # Fallback: use answer as rule
                rules.append({
                    "type": "learning_session",
                    "source": "Q&A pair",
                    "rule": answer[:200],  # First 200 chars
                    "question": question,
                    "answer": answer,
                    "date": pair.get("date", datetime.now().isoformat()),
                    "fallback": True
                })
        
        return rules
    
    def get_behavior_rules(self) -> List[Dict]:
        """Get behavior rules as-is (no summarization needed)"""
        behavioral_data = self.data_manager.get_behavioral_rules()
        rules = []
        
        for behavior in behavioral_data.get("behaviors", []):
            rules.append({
                "type": "behavior",
                "source": behavior.get("category", "General"),
                "rule": behavior.get("description", ""),
                "date": datetime.now().isoformat()
            })
        
        return rules
    
    def generate_all_rules(self, force_regenerate: bool = False, additive: bool = False) -> List[Dict]:
        """Generate all rules from all sources
        
        Args:
            force_regenerate: If True, regenerate all rules from scratch
            additive: If True, only add rules for new content that doesn't already have rules
        """
        existing_rules = []
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r') as f:
                    cached_rules = json.load(f)
                    existing_rules = cached_rules.get("rules", [])
            except:
                pass
        
        # If additive mode, check what already exists
        if additive and existing_rules:
            # Build sets of existing sources to avoid duplicates
            existing_context_sources = set()
            existing_learning_sources = set()
            existing_behavior_sources = set()
            
            for rule in existing_rules:
                rule_type = rule.get("type", "")
                source = rule.get("source", "")
                if rule_type == "context":
                    existing_context_sources.add(source)
                elif rule_type == "learning_session":
                    # For learning sessions, use question as identifier
                    question = rule.get("question", "")
                    if question:
                        existing_learning_sources.add(question)
                elif rule_type == "behavior":
                    # For behavior rules, use description as identifier
                    desc = rule.get("rule", "")
                    if desc:
                        existing_behavior_sources.add(desc)
            
            # Generate new rules
            new_rules = []
            
            # Context rules - only for new files
            context_files = self.data_manager.get_all_context_files()
            for context_item in context_files:
                filename = context_item["metadata"].get("filename", "unknown")
                if filename not in existing_context_sources:
                    # Generate rule for this new context file
                    text = context_item["text"]
                    metadata = context_item["metadata"]
                    
                    summary_prompt = f"""Based on the following context, create a concise rule or guideline that captures the key information. The rule should be actionable and specific.

Context:
{text[:2000]}

Create a brief rule (1-3 sentences) that summarizes the key information from this context:"""
                    
                    try:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant that creates concise, actionable rules from context."},
                            {"role": "user", "content": summary_prompt}
                        ]
                        summary = self.ollama_client.chat(messages, stream=False)
                        
                        new_rules.append({
                            "type": "context",
                            "source": filename,
                            "rule": summary.strip(),
                            "metadata": metadata,
                            "date": datetime.now().isoformat()
                        })
                    except Exception as e:
                        sentences = text.split('.')[:3]
                        fallback_rule = '. '.join(s for s in sentences if s.strip()) + '.'
                        new_rules.append({
                            "type": "context",
                            "source": filename,
                            "rule": fallback_rule,
                            "metadata": metadata,
                            "date": datetime.now().isoformat(),
                            "fallback": True
                        })
            
            # Learning session rules - only for new pairs
            learned_pairs = self.data_manager.get_learned_pairs()
            for pair in learned_pairs:
                question = pair.get("question", "")
                if question and question not in existing_learning_sources:
                    answer = pair.get("answer", "")
                    
                    rule_prompt = f"""Based on this question and answer, create a concise rule or guideline:

Question: {question}
Answer: {answer}

Create a brief rule (1-2 sentences) that captures the key principle or information:"""
                    
                    try:
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant that creates concise rules from Q&A pairs."},
                            {"role": "user", "content": rule_prompt}
                        ]
                        rule_text = self.ollama_client.chat(messages, stream=False)
                        
                        new_rules.append({
                            "type": "learning_session",
                            "source": "Q&A pair",
                            "rule": rule_text.strip(),
                            "question": question,
                            "answer": answer,
                            "date": pair.get("date", datetime.now().isoformat())
                        })
                    except Exception as e:
                        new_rules.append({
                            "type": "learning_session",
                            "source": "Q&A pair",
                            "rule": answer[:200],
                            "question": question,
                            "answer": answer,
                            "date": pair.get("date", datetime.now().isoformat()),
                            "fallback": True
                        })
            
            # Behavior rules - check for new ones
            behavioral_data = self.data_manager.get_behavioral_rules()
            for behavior in behavioral_data.get("behaviors", []):
                desc = behavior.get("description", "")
                if desc and desc not in existing_behavior_sources:
                    new_rules.append({
                        "type": "behavior",
                        "source": behavior.get("category", "General"),
                        "rule": desc,
                        "date": datetime.now().isoformat()
                    })
            
            # Combine existing and new rules
            all_rules = existing_rules + new_rules
            
            # Save updated rules
            self.rules_dir.mkdir(parents=True, exist_ok=True)
            with open(self.rules_file, 'w') as f:
                json.dump({
                    "rules": all_rules,
                    "last_updated": datetime.now().isoformat(),
                    "model_name": self.model_name
                }, f, indent=2)
            
            return all_rules
        
        # Check if rules already exist and are recent (only if not forcing regenerate)
        if not force_regenerate and self.rules_file.exists():
            try:
                with open(self.rules_file, 'r') as f:
                    cached_rules = json.load(f)
                    # Check if cache is recent (within last hour)
                    if cached_rules.get("last_updated"):
                        last_update = datetime.fromisoformat(cached_rules["last_updated"])
                        if (datetime.now() - last_update).total_seconds() < 3600:
                            return cached_rules.get("rules", [])
            except:
                pass
        
        # Generate all rules from scratch
        all_rules = []
        
        # Context rules
        all_rules.extend(self.generate_context_summaries())
        
        # Learning session rules
        all_rules.extend(self.generate_learning_session_summaries())
        
        # Behavior rules (no summarization)
        all_rules.extend(self.get_behavior_rules())
        
        # Save rules
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        with open(self.rules_file, 'w') as f:
            json.dump({
                "rules": all_rules,
                "last_updated": datetime.now().isoformat(),
                "model_name": self.model_name
            }, f, indent=2)
        
        return all_rules
    
    def get_all_rules(self) -> List[Dict]:
        """Get all rules from cache (without regenerating)"""
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r') as f:
                    cached_rules = json.load(f)
                    return cached_rules.get("rules", [])
            except:
                return []
        return []
    
    def update_rule(self, rule_index: int, updated_rule: Dict) -> bool:
        """Update a rule at the given index"""
        try:
            rules = self.get_all_rules()
            if 0 <= rule_index < len(rules):
                # Update the rule
                updated_rule["date"] = datetime.now().isoformat()  # Update date
                rules[rule_index] = updated_rule
                
                # Save updated rules
                with open(self.rules_file, 'w') as f:
                    json.dump({
                        "rules": rules,
                        "last_updated": datetime.now().isoformat(),
                        "model_name": self.model_name
                    }, f, indent=2)
                return True
        except Exception as e:
            print(f"Error updating rule: {e}")
        return False
    
    def delete_rule(self, rule_index: int) -> bool:
        """Delete a rule at the given index"""
        try:
            rules = self.get_all_rules()
            if 0 <= rule_index < len(rules):
                # Remove the rule
                rules.pop(rule_index)
                
                # Save updated rules
                with open(self.rules_file, 'w') as f:
                    json.dump({
                        "rules": rules,
                        "last_updated": datetime.now().isoformat(),
                        "model_name": self.model_name
                    }, f, indent=2)
                return True
        except Exception as e:
            print(f"Error deleting rule: {e}")
        return False
    
    def find_relevant_rules(self, user_prompt: str, max_rules: int = 5) -> List[Dict]:
        """Find rules relevant to the user's prompt, prioritizing context files by relevance"""
        # Load or generate rules
        rules = self.get_all_rules()  # Use get_all_rules to avoid regeneration
        
        if not rules:
            return []
        
        # Separate rules by type for prioritized search
        context_rules = [r for r in rules if r.get("type") == "context"]
        learning_rules = [r for r in rules if r.get("type") == "learning_session"]
        behavior_rules = [r for r in rules if r.get("type") == "behavior"]
        other_rules = [r for r in rules if r.get("type") not in ["context", "learning_session", "behavior"]]
        
        prioritized_rules = []
        
        # Step 1: Find most relevant context files first (prioritize these)
        if context_rules:
            try:
                context_rules_text = "\n\n".join([
                    f"Context {i+1} ({rule.get('source', 'file')}): {rule['rule'][:300]}..." 
                    if len(rule.get('rule', '')) > 300 
                    else f"Context {i+1} ({rule.get('source', 'file')}): {rule['rule']}"
                    for i, rule in enumerate(context_rules)
                ])
                
                context_relevance_prompt = f"""Given the user's prompt below, identify which context files are most relevant, ordered by relevance (most relevant first).
Return only the context numbers in order of relevance, separated by commas (e.g., "2, 1, 3" means context 2 is most relevant).

User Prompt: {user_prompt}

Available Context Files:
{context_rules_text}

Which context file numbers are most relevant to the user's prompt, in order of relevance? Return only numbers separated by commas:"""
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that identifies and ranks relevant context files for user prompts. Return context file numbers in order of relevance, separated by commas."},
                    {"role": "user", "content": context_relevance_prompt}
                ]
                response = self.ollama_client.chat(messages, stream=False)
                
                # Parse response to get context file numbers in relevance order
                context_numbers = []
                for part in response.split(','):
                    try:
                        num = int(part.strip())
                        if 1 <= num <= len(context_rules):
                            rule_idx = num - 1
                            if rule_idx not in context_numbers:
                                context_numbers.append(rule_idx)
                    except:
                        continue
                
                # Add context files in relevance order (prioritized)
                for idx in context_numbers:
                    if len(prioritized_rules) < max_rules:
                        prioritized_rules.append(context_rules[idx])
            except Exception as e:
                # Fallback: add all context files if relevance check fails
                for rule in context_rules[:max_rules]:
                    prioritized_rules.append(rule)
        
        # Step 2: Find other relevant rules (learning sessions, behavior, etc.)
        remaining_slots = max_rules - len(prioritized_rules)
        if remaining_slots > 0:
            other_rules_list = learning_rules + behavior_rules + other_rules
            if other_rules_list:
                try:
                    other_rules_text = "\n\n".join([
                        f"Rule {i+1} ({rule['type']}): {rule['rule'][:200]}..." 
                        if len(rule.get('rule', '')) > 200 
                        else f"Rule {i+1} ({rule['type']}): {rule['rule']}"
                        for i, rule in enumerate(other_rules_list)
                    ])
                    
                    other_relevance_prompt = f"""Given the user's prompt below, identify which rules are most relevant, ordered by relevance (most relevant first).
Return only the rule numbers in order of relevance, separated by commas.

User Prompt: {user_prompt}

Available Rules:
{other_rules_text}

Which rule numbers are most relevant to the user's prompt, in order of relevance? Return only numbers separated by commas:"""
                    
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that identifies and ranks relevant rules for user prompts. Return rule numbers in order of relevance, separated by commas."},
                        {"role": "user", "content": other_relevance_prompt}
                    ]
                    response = self.ollama_client.chat(messages, stream=False)
                    
                    # Parse response to get rule numbers in order
                    rule_numbers = []
                    for part in response.split(','):
                        try:
                            num = int(part.strip())
                            if 1 <= num <= len(other_rules_list):
                                rule_idx = num - 1
                                if rule_idx not in rule_numbers:
                                    rule_numbers.append(rule_idx)
                        except:
                            continue
                    
                    # Add other rules in relevance order
                    for idx in rule_numbers:
                        if len(prioritized_rules) < max_rules:
                            prioritized_rules.append(other_rules_list[idx])
                except Exception as e:
                    # Fallback: add other rules if relevance check fails
                    for rule in other_rules_list:
                        if len(prioritized_rules) < max_rules:
                            prioritized_rules.append(rule)
        
        # If we still have slots and no rules were found via relevance, use fallback
        if not prioritized_rules:
            # Fallback: prioritize context files, then others
            for rule in context_rules[:max_rules]:
                prioritized_rules.append(rule)
            remaining = max_rules - len(prioritized_rules)
            if remaining > 0:
                for rule in (learning_rules + behavior_rules + other_rules)[:remaining]:
                    prioritized_rules.append(rule)
        
        return prioritized_rules
    
    def inject_rules_into_prompt(self, user_prompt: str, max_rules: int = 5, max_context_length: int = 8000, max_total_context: int = 20000) -> str:
        """Inject relevant rules into the user prompt, prioritizing context files by relevance
        
        Args:
            user_prompt: The original user prompt
            max_rules: Maximum number of rules to inject
            max_context_length: Maximum characters per context file (to prevent overly long prompts)
            max_total_context: Maximum total characters for all context files combined
        """
        relevant_rules = self.find_relevant_rules(user_prompt, max_rules)
        
        if not relevant_rules:
            return user_prompt
        
        # Separate rules by type for different sections
        context_parts = []
        rules_parts = []  # For behavioral rules and learning sessions
        total_context_length = 0
        context_files_added = 0
        max_context_files = 3  # Limit number of context files to prevent prompt bloat
        
        # Process rules in order (already prioritized by find_relevant_rules)
        for rule in relevant_rules:
            rule_type = rule.get("type", "")
            
            if rule_type == "context":
                # Prioritize context files, but limit total size and count
                if context_files_added >= max_context_files:
                    continue  # Skip if we've already added enough context files
                
                metadata = rule.get("metadata", {})
                text_file_name = metadata.get("text_file", "")
                
                if text_file_name:
                    # Try to load the full context text
                    context_dir = get_model_context_dir(self.model_name)
                    text_file_path = context_dir / text_file_name
                    
                    if text_file_path.exists():
                        try:
                            with open(text_file_path, 'r', encoding='utf-8') as f:
                                full_text = f.read()
                            
                            # Calculate how much we can add without exceeding total limit
                            remaining_budget = max_total_context - total_context_length
                            
                            if remaining_budget > 0:
                                # Truncate if needed (prioritize beginning of file)
                                if len(full_text) > remaining_budget:
                                    # Use remaining budget, but cap at max_context_length per file
                                    available_length = min(remaining_budget, max_context_length)
                                    full_text = full_text[:available_length] + "\n[... content truncated ...]"
                                elif len(full_text) > max_context_length:
                                    # File exceeds per-file limit but fits in total budget
                                    full_text = full_text[:max_context_length] + "\n[... content truncated ...]"
                                
                                context_parts.append(f"**Context from {metadata.get('filename', 'file')}:**\n{full_text}")
                                total_context_length += len(full_text)
                                context_files_added += 1
                            else:
                                # No more budget for context files, skip this one
                                continue
                        except Exception as e:
                            # Fallback to summary if file can't be read
                            context_parts.append(f"**Context Rule:** {rule['rule']}")
                    else:
                        # Fallback to summary if file doesn't exist
                        context_parts.append(f"**Context Rule:** {rule['rule']}")
                else:
                    # Fallback to summary if no text_file in metadata
                    context_parts.append(f"**Context Rule:** {rule['rule']}")
            
            elif rule_type == "learning_session":
                # For learning session rules, include both question and answer for full context
                question = rule.get("question", "")
                answer = rule.get("answer", "")
                if question and answer:
                    rules_parts.append(f"**Learning from Q&A:**\nQ: {question}\nA: {answer}")
                else:
                    # Fallback to just the rule summary
                    rules_parts.append(f"**Learning Rule:** {rule['rule']}")
            
            elif rule_type == "behavior":
                # For behavior rules, use the rule text as-is
                rule_source = rule.get("source", "")
                rule_text = rule.get("rule", "")
                if rule_source:
                    rules_parts.append(f"**Behavior Rule ({rule_source}):** {rule_text}")
                else:
                    rules_parts.append(f"**Behavior Rule:** {rule_text}")
            
            else:
                # For other rule types
                rule_label = f"**{rule_type.title()} Rule:**" if rule_type else "**Rule:**"
                rules_parts.append(f"{rule_label} {rule['rule']}")
        
        # Build the enhanced prompt with the new structure
        enhanced_prompt_parts = []
        
        # Section 1: Background Information (Context files)
        if context_parts:
            context_text = "\n\n".join(context_parts)
            enhanced_prompt_parts.append(f"BACKGROUND INFORMATION FOR CONTEXT ONLY:\n{context_text}")
        
        # Section 2: Additional Rules (Behavioral rules and learning sessions)
        if rules_parts:
            rules_text = "\n\n".join(rules_parts)
            enhanced_prompt_parts.append(f"ADDITIONAL RULES FOR ANSWERING:\n{rules_text}")
        
        # Section 3: The actual question
        enhanced_prompt_parts.append(f"HERE IS THE QUESTION:\n{user_prompt}")
        
        # Combine all sections
        enhanced_prompt = "\n\n".join(enhanced_prompt_parts)
        
        # Add instructions if we have context or rules
        if context_parts or rules_parts:
            enhanced_prompt += "\n\nInstructions: When answering the question above, use the background information and additional rules to help you in your answer where relevant, but be sure to remain focused on the question and do not stray off that topic. Don't mistake the background information or rules for the question. Just use them to help answer, where relevant, the question above."
        
        return enhanced_prompt

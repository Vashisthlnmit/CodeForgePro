import streamlit as st
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import io
# Configuration classes
class State(TypedDict):
    topic: str
    user_stories: str
    product_feedback: str
    product_approval: str
    design_document: str
    design_approval: str
    design_feedback: str
    code: str
    code_feedback: str
    code_approval: str
    security_review: str
    security_approval: str
    security_feedback: str
    test_case: str
    test_case_feedback: str
    test_case_approval: str
    question_answer_testing: str
    question_answer_testing_feedback: str
    question_answer_testing_approval: str
    final_code_docs: str

class ProductFeedback(BaseModel):
    product_feedback: str = Field(description="Feedback to improve the description")
    product_approval: Literal["Approved", "FeedBack"] = Field(description="Approval status")

class DesignFeedback(BaseModel):
    design_feedback: str = Field(description="Feedback to improve the design")
    design_approval: Literal["Approved", "FeedBack"] = Field(description="Approval status")

class CodeFeedback(BaseModel):
    code_feedback: str = Field(description="Feedback on the code")
    code_approval: Literal["Approved", "FeedBack"] = Field(description="Approval status")

class SecurityFeedback(BaseModel):
    security_review: str = Field(description="Security feedback")
    security_approval: Literal["Approved", "FeedBack"] = Field(description="Approval status")

class TestcaseFeedback(BaseModel):
    test_case_feedback: str = Field(description="Test case feedback")
    test_case_approval: Literal["Approved", "FeedBack"] = Field(description="Approval status")

class QuestionAnswerFeedback(BaseModel):
    question_answer_testing_feedback: str = Field(description="QA testing feedback")
    question_answer_testing_approval: Literal["Passed", "Failed"] = Field(description="Approval status")

# Initialize models using session state
def initialize_model(Type: str, model_name: str, api_key: str):
    try:
        if Type.lower() == "groq":
            st.session_state.model = ChatGroq(model_name=model_name, api_key=api_key)
        else:
            st.session_state.model = OpenAI(model_name=model_name, api_key=api_key)
        
        # Initialize all planners
        st.session_state.Product_Planner = create_structured_planner(st.session_state.model, ProductFeedback)
        st.session_state.Design_Planner = create_structured_planner(st.session_state.model, DesignFeedback)
        st.session_state.Code_Planner = create_structured_planner(st.session_state.model, CodeFeedback)
        st.session_state.Security_Planner = create_structured_planner(st.session_state.model, SecurityFeedback)
        st.session_state.TestCase_Planner = create_structured_planner(st.session_state.model, TestcaseFeedback)
        st.session_state.QuestionAnswer_Planner = create_structured_planner(st.session_state.model, QuestionAnswerFeedback)
        return True
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return False

def create_structured_planner(model, output_class):
    parser = JsonOutputParser(pydantic_object=output_class)
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following content and provide feedback in JSON format:
    {input}
    
    {format_instructions}
    """)
    return prompt | model | parser

# Content generation functions
def Content_Writer(state: State):
    if 'model' not in st.session_state:
        raise ValueError("Model not initialized")
    
    prompt = ChatPromptTemplate.from_template("""
    {instruction}
    
    Topic: {topic}
    {feedback}
    """)
    
    if state.get("product_feedback"):
        instruction = "Improve this description based on the feedback:"
        feedback = f"Feedback: {state['product_feedback']}"
    else:
        instruction = "Create a detailed description for:"
        feedback = ""
    
    chain = prompt | st.session_state.model | StrOutputParser()
    msg = chain.invoke({
        "instruction": instruction,
        "topic": state["topic"],
        "feedback": feedback
    })
    return {"user_stories": msg}

def Design_Engineer(state: State):
    if 'model' not in st.session_state:
        raise ValueError("Model not initialized")
    
    prompt = ChatPromptTemplate.from_template("""
    Create a technical design document based on these user stories:
    {user_stories}
    
    {feedback}
    """)
    
    if state.get("design_feedback"):
        feedback = f"Design Feedback: {state['design_feedback']}"
    else:
        feedback = ""
    
    chain = prompt | st.session_state.model | StrOutputParser()
    msg = chain.invoke({
        "user_stories": state["user_stories"],
        "feedback": feedback
    })
    return {"design_document": msg}

def Coder(state: State):
    if 'model' not in st.session_state:
        raise ValueError("Model not initialized")
    
    prompt = ChatPromptTemplate.from_template("""
    Write code implementation for this design:
    {design_document}
    
    {feedback}
    """)
    
    if state.get("code_feedback"):
        feedback = f"Code Feedback: {state['code_feedback']}"
    else:
        feedback = ""
    
    chain = prompt | st.session_state.model | StrOutputParser()
    msg = chain.invoke({
        "design_document": state["design_document"],
        "feedback": feedback
    })
    return {"code": msg}

def Tester(state: State):
    if 'model' not in st.session_state:
        raise ValueError("Model not initialized")
    
    prompt = ChatPromptTemplate.from_template("""
    Create test cases for this code:
    {code}
    
    {feedback}
    """)
    
    if state.get("test_case_feedback"):
        feedback = f"Test Case Feedback: {state['test_case_feedback']}"
    else:
        feedback = ""
    
    chain = prompt | st.session_state.model | StrOutputParser()
    msg = chain.invoke({
        "code": state["code"],
        "feedback": feedback
    })
    return {"test_case": msg}

def QATester(state: State):
    if 'model' not in st.session_state:
        raise ValueError("Model not initialized")
    
    prompt = ChatPromptTemplate.from_template("""
    Create QA testing scenarios for these test cases:
    {test_case}
    
    {feedback}
    """)
    
    if state.get("question_answer_testing_feedback"):
        feedback = f"QA Feedback: {state['question_answer_testing_feedback']}"
    else:
        feedback = ""
    
    chain = prompt | st.session_state.model | StrOutputParser()
    msg = chain.invoke({
        "test_case": state["test_case"],
        "feedback": feedback
    })
    return {"question_answer_testing": msg}

def SecurityReview(state: State):
    if 'model' not in st.session_state:
        raise ValueError("Model not initialized")
    
    prompt = ChatPromptTemplate.from_template("""
    Analyze this code for security issues:
    {code}
    
    {feedback}
    """)
    
    if state.get("security_review"):
        feedback = f"Security Feedback: {state['security_review']}"
    else:
        feedback = ""
    
    chain = prompt | st.session_state.model | StrOutputParser()
    msg = chain.invoke({
        "code": state["code"],
        "feedback": feedback
    })
    return {"security_review": msg}

# Review functions
def Product_Owner_Review(state: State):
    if 'Product_Planner' not in st.session_state:
        raise ValueError("Product Planner not initialized")
    
    response = st.session_state.Product_Planner.invoke({
        "input": f"""
        Compare the original requirements with the generated user stories:
        
        Requirements: {state['topic']}
        User Stories: {state['user_stories']}
        """,
        "format_instructions": JsonOutputParser(pydantic_object=ProductFeedback).get_format_instructions()
    })
    return {
        "product_approval": response["product_approval"],
        "product_feedback": response["product_feedback"]
    }

def Design_Owner_Review(state: State):
    if 'Design_Planner' not in st.session_state:
        raise ValueError("Design Planner not initialized")
    
    response = st.session_state.Design_Planner.invoke({
        "input": f"""
        Review this design document against requirements:
        
        Requirements: {state['topic']}
        Design: {state['design_document']}
        """,
        "format_instructions": JsonOutputParser(pydantic_object=DesignFeedback).get_format_instructions()
    })
    return {
        "design_approval": response["design_approval"],
        "design_feedback": response["design_feedback"]
    }

def Code_Owner_Review(state: State):
    if 'Code_Planner' not in st.session_state:
        raise ValueError("Code Planner not initialized")
    
    response = st.session_state.Code_Planner.invoke({
        "input": f"""
        Review this code against the design:
        
        Design: {state['design_document']}
        Code: {state['code']}
        """,
        "format_instructions": JsonOutputParser(pydantic_object=CodeFeedback).get_format_instructions()
    })
    return {
        "code_approval": response["code_approval"],
        "code_feedback": response["code_feedback"]
    }

def Security_Owner_Review(state: State):
    if 'Security_Planner' not in st.session_state:
        raise ValueError("Security Planner not initialized")
    
    response = st.session_state.Security_Planner.invoke({
        "input": f"""
        Review this security analysis:
        
        Code: {state['code']}
        Security Review: {state['security_review']}
        """,
        "format_instructions": JsonOutputParser(pydantic_object=SecurityFeedback).get_format_instructions()
    })
    return {
        "security_approval": response["security_approval"],
        "security_feedback": response["security_review"]
    }

def TestCase_Owner_Review(state: State):
    if 'TestCase_Planner' not in st.session_state:
        raise ValueError("Test Case Planner not initialized")
    
    response = st.session_state.TestCase_Planner.invoke({
        "input": f"""
        Review these test cases against the code:
        
        Code: {state['code']}
        Test Cases: {state['test_case']}
        """,
        "format_instructions": JsonOutputParser(pydantic_object=TestcaseFeedback).get_format_instructions()
    })
    return {
        "test_case_approval": response["test_case_approval"],
        "test_case_feedback": response["test_case_feedback"]
    }

def Question_Owner_Review(state: State):
    if 'QuestionAnswer_Planner' not in st.session_state:
        raise ValueError("Question Answer Planner not initialized")
    
    response = st.session_state.QuestionAnswer_Planner.invoke({
        "input": f"""
        Review these QA test scenarios:
        
        Test Cases: {state['test_case']}
        QA Testing: {state['question_answer_testing']}
        """,
        "format_instructions": JsonOutputParser(pydantic_object=QuestionAnswerFeedback).get_format_instructions()
    })
    return {
        "question_answer_testing_approval": response["question_answer_testing_approval"],
        "question_answer_testing_feedback": response["question_answer_testing_feedback"]
    }

# Routing functions
def product_route(state: State):
    if state['product_approval'] == "Approved":
        return "Approved"
    return "FeedBack"

def design_route(state: State):
    if state["design_approval"] == "Approved":
        return "Approved"
    return "FeedBack"

def code_route(state: State):
    if state["code_approval"] == "Approved":
        return "Approved"
    return "FeedBack"

def security_route(state: State):
    if state["security_approval"] == "Approved":
        return "Approved"
    return "FeedBack"

def test_route(state: State):
    if state["test_case_approval"] == "Approved":
        return "Approved"
    return "FeedBack"

def question_route(state: State):
    if state["question_answer_testing_approval"] == "Passed":
        return "Passed"
    return "Failed"

# Streamlit UI
def main():
    st.title("Software Development Workflow Automation")
    
    # Initialize session state
    if 'state' not in st.session_state:
        st.session_state.state = State(
            topic='',
            user_stories='',
            product_feedback='',
            product_approval='',
            design_document='',
            design_approval='',
            design_feedback='',
            code='',
            code_feedback='',
            code_approval='',
            security_review='',
            security_approval='',
            security_feedback='',
            test_case='',
            test_case_feedback='',
            test_case_approval='',
            question_answer_testing='',
            question_answer_testing_feedback='',
            question_answer_testing_approval='',
            final_code_docs=''
        )
    
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False

    # Sidebar configuration
    with st.sidebar:
        st.header("Configurations")
        selected_LLM = st.selectbox("Select LLM", ("Groq", "OpenAI"))
        
        if selected_LLM == "Groq":
            selected_Model = st.selectbox("Select Model", 
                ("llama3-70b-8192", "gemma2-9b-it"))
            Api_Key = st.text_input("Enter Groq API Key", type="password")
        else:
            selected_Model = st.selectbox("Select Model",
                ("GPT-3.5 Turbo", "GPT-4 Turbo", "GPT-4"))
            Api_Key = st.text_input("Enter OpenAI API Key", type="password")
        
        if st.button("Initialize Models"):
            if not Api_Key:
                st.error("Please enter your API key")
            else:
                if initialize_model(selected_LLM, selected_Model, Api_Key):
                    st.session_state.model_initialized = True
                    st.success("Model initialized successfully!")

    # Main interface
    st.session_state.state['topic'] = st.text_area(
        "Enter your project requirements:",
        value=st.session_state.state['topic'],
        height=150
    )
    
    if st.button("Run Workflow", disabled=not st.session_state.model_initialized):
        if not st.session_state.model_initialized:
            st.error("Please initialize the model first")
        else:
            try:
                # Build workflow graph
                workflow = StateGraph(State)
                
                # Add nodes
                workflow.add_node("content_writer", Content_Writer)
                workflow.add_node("product_owner_review", Product_Owner_Review)
                workflow.add_node("design_engineer", Design_Engineer)
                workflow.add_node("design_owner_review", Design_Owner_Review)
                workflow.add_node("coder", Coder)
                workflow.add_node("code_owner_review", Code_Owner_Review)
                workflow.add_node("security_review_check", SecurityReview)
                workflow.add_node("security_owner_review", Security_Owner_Review)
                workflow.add_node("tester", Tester)
                workflow.add_node("test_owner_review", TestCase_Owner_Review)
                workflow.add_node("qa_tester", QATester)
                workflow.add_node("question_owner_review", Question_Owner_Review)
                
                # Add edges
                workflow.add_edge(START, "content_writer")
                workflow.add_edge("content_writer", "product_owner_review")
                
                workflow.add_conditional_edges(
                    "product_owner_review",
                    product_route,
                    {"Approved": "design_engineer", "FeedBack": "content_writer"}
                )
                
                workflow.add_edge("design_engineer", "design_owner_review")
                workflow.add_conditional_edges(
                    "design_owner_review",
                    design_route,
                    {"Approved": "coder", "FeedBack": "design_engineer"}
                )
                
                workflow.add_edge("coder", "code_owner_review")
                workflow.add_conditional_edges(
                    "code_owner_review",
                    code_route,
                    {"Approved": "security_review_check", "FeedBack": "coder"}
                )
                
                workflow.add_edge("security_review_check", "security_owner_review")
                workflow.add_conditional_edges(
                    "security_owner_review",
                    security_route,
                    {"Approved": "tester", "FeedBack": "coder"}
                )
                
                workflow.add_edge("tester", "test_owner_review")
                workflow.add_conditional_edges(
                    "test_owner_review",
                    test_route,
                    {"Approved": "qa_tester", "FeedBack": "tester"}
                )
                
                workflow.add_edge("qa_tester", "question_owner_review")
                workflow.add_conditional_edges(
                    "question_owner_review",
                    question_route,
                    {"Passed": END, "Failed": "qa_tester"}
                )
                
                # Compile and run workflow
                app = workflow.compile()
                
                with st.spinner("Executing workflow..."):
                    for step in app.stream(st.session_state.state,{"recursion_limit": 100}):
                        for key, value in step.items():
                            if key != "__end__":
                                st.session_state.state.update(value)
                
                # Display results
                st.success("Workflow completed!")
                
                st.subheader("Final Workflow")
                cols = st.columns(1)
                
                with cols[0]:
                    st.write("**User Stories**")
                    st.write(st.session_state.state['user_stories'])
                    
                    st.write("**Design Document**")
                    st.write(st.session_state.state['design_document'])
                    
                    st.write("**Generated Code**")
                    st.code(st.session_state.state['code'], language='python')

                    st.write("**Test Cases**")
                    st.write(st.session_state.state['test_case'])
                    
                    st.write("**QA Testing**")
                    st.write(st.session_state.state['question_answer_testing'])
                    
                    st.write("**Security Review**")
                    st.write(st.session_state.state['security_review'])
                    txt_content=f"User_Stories\n {st.session_state.state['user_stories']}\n\nDesign Document\n {st.session_state.state['design_document']}\n\nGenerated Code\n {st.session_state.state['code']}\n\nTest Cases\n {st.session_state.state['test_case']}\n\nQA Testing\n {st.session_state.state['question_answer_testing']}\n\nSecurity Review\n {st.session_state.state['security_review']}"   
                    txt_file=io.BytesIO(txt_content.encode())
                    st.download_button(label="Download TXT",data=txt_file,file_name="complete_code_report",mime="text/plain")
            except Exception as e:
                st.error(f"Workflow failed: {str(e)}")
if __name__ == "__main__":
    main()
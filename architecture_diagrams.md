# Test Driver Architecture Diagrams

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "User Layer"
        U1[IDE/Editor with MCP Support]
        U2[CLI Tool]
        U3[CI/CD Pipeline]
    end

    subgraph "Test Driver MCP Server"
        MCP[MCP Server Core<br/>JSON-RPC 2.0]
        
        subgraph "Core Services"
            ATE[Autonomous Testing Engine]
            TPG[Test Plan Generator]
            TER[Test Execution Runner]
            REP[Report Generator]
        end
        
        subgraph "Vision Layer"
            VM[Vision Manager]
            VC[Vision Cache]
        end
        
        subgraph "Execution Layer"
            EM[Execution Manager]
            SM[Session Manager]
        end
        
        subgraph "Storage"
            TS[Test Scripts Store]
            RS[Results Store]
            AS[Artifacts Store]
        end
    end

    subgraph "Vision Adapters"
        VA1[OpenAI GPT-4V]
        VA2[Anthropic Claude]
        VA3[Google Gemini]
        VA4[Local VLM<br/>Qwen/Llama]
        VA5[Custom Model]
    end

    subgraph "Execution Adapters"
        EA1[Selenium WebDriver]
        EA2[Playwright]
    end

    subgraph "Browser Layer"
        B1[Chrome/Chromium]
        B2[Firefox]
        B3[WebKit/Safari]
        B4[Edge]
    end

    U1 --> MCP
    U2 --> MCP
    U3 --> MCP

    MCP --> ATE
    ATE --> TPG
    ATE --> TER
    ATE --> REP

    TER --> VM
    TER --> EM

    VM --> VC
    VM --> VA1
    VM --> VA2
    VM --> VA3
    VM --> VA4
    VM --> VA5

    EM --> SM
    EM --> EA1
    EM --> EA2

    EA1 --> B1
    EA1 --> B2
    EA1 --> B3
    EA1 --> B4

    EA2 --> B1
    EA2 --> B2
    EA2 --> B3

    ATE --> TS
    TER --> RS
    TER --> AS

    style MCP fill:#4A90E2
    style ATE fill:#50C878
    style VM fill:#FFB347
    style EM fill:#FF6B6B
```

## 2. Component Interaction Sequence

```mermaid
sequenceDiagram
    participant User
    participant MCPHost as MCP Host
    participant MCPServer as MCP Server
    participant ATE as Autonomous Engine
    participant Vision as Vision Module
    participant Execution as Execution Engine
    participant Browser

    User->>MCPHost: Provide test requirements
    MCPHost->>MCPServer: testdriver.startTest(requirements)
    MCPServer->>ATE: Initialize test session
    
    ATE->>ATE: Generate test plan from requirements
    ATE->>MCPServer: Return test plan
    MCPServer->>MCPHost: Display test plan for approval
    MCPHost->>User: Show test plan
    User->>MCPHost: Approve test plan
    
    loop For each test step
        MCPHost->>MCPServer: Execute next step
        MCPServer->>ATE: Process step
        ATE->>Execution: Navigate/Get screenshot
        Execution->>Browser: Perform action
        Browser-->>Execution: Screenshot
        Execution-->>ATE: Return screenshot
        
        ATE->>Vision: Analyze screenshot + find element
        Vision->>Vision: OCR + Element detection
        Vision-->>ATE: Element coordinates + description
        
        ATE->>Execution: Perform action at coordinates
        Execution->>Browser: Click/Type/Scroll
        Browser-->>Execution: Action result
        Execution-->>ATE: Execution status
        
        ATE->>Execution: Verify result
        Execution->>Browser: Get new screenshot
        Browser-->>Execution: Screenshot
        Execution-->>ATE: Verification screenshot
        
        ATE->>Vision: Verify expected state
        Vision-->>ATE: Verification result
        
        ATE->>MCPServer: Step complete + artifacts
        MCPServer->>MCPHost: Update progress
    end
    
    ATE->>ATE: Generate comprehensive report
    ATE->>MCPServer: Test complete + report
    MCPServer->>MCPHost: testdriver.getReport()
    MCPHost->>User: Display report
```

## 3. Vision Module Architecture

```mermaid
graph TD
    subgraph "Vision Module"
        VI[Vision Interface]
        
        subgraph "Processing Pipeline"
            PP1[Image Preprocessor]
            PP2[OCR Engine]
            PP3[Element Detector]
            PP4[Semantic Analyzer]
        end
        
        subgraph "Adapter Layer"
            AL[Adapter Manager]
            AC[Adapter Config]
        end
        
        subgraph "Caching Layer"
            CL1[Element Cache]
            CL2[Screen Hash Cache]
            CL3[Coordinate Cache]
        end
    end

    subgraph "Vision Adapters"
        subgraph "Cloud Adapters"
            CA1[GPT-4V Adapter]
            CA2[Claude Adapter]
            CA3[Gemini Adapter]
        end
        
        subgraph "Local Adapters"
            LA1[Ollama Adapter]
            LA2[HuggingFace Adapter]
            LA3[ONNX Runtime Adapter]
        end
        
        subgraph "Specialized Adapters"
            SA1[Tesseract OCR]
            SA2[EasyOCR]
            SA3[PaddleOCR]
        end
    end

    VI --> PP1
    PP1 --> PP2
    PP1 --> PP3
    PP1 --> PP4

    PP2 --> AL
    PP3 --> AL
    PP4 --> AL

    AL --> AC
    AL --> CL1
    AL --> CL2
    AL --> CL3

    AL --> CA1
    AL --> CA2
    AL --> CA3
    AL --> LA1
    AL --> LA2
    AL --> LA3
    AL --> SA1
    AL --> SA2
    AL --> SA3

    style VI fill:#FFB347
    style AL fill:#9370DB
    style CL1 fill:#87CEEB
```

## 4. Execution Engine Architecture

```mermaid
graph TD
    subgraph "Execution Engine"
        EI[Execution Interface]
        
        subgraph "Command Layer"
            CL1[Navigation Commands]
            CL2[Interaction Commands]
            CL3[Assertion Commands]
            CL4[Utility Commands]
        end
        
        subgraph "Session Management"
            SM1[Session Pool]
            SM2[Browser Context Manager]
            SM3[Cookie/Storage Manager]
        end
        
        subgraph "Monitoring"
            M1[Network Monitor]
            M2[Console Logger]
            M3[Performance Tracker]
            M4[Video Recorder]
        end
    end

    subgraph "Framework Adapters"
        subgraph "Selenium Adapter"
            SE1[WebDriver Manager]
            SE2[Element Locator]
            SE3[Wait Strategy]
            SE4[Screenshot Handler]
        end
        
        subgraph "Playwright Adapter"
            PW1[Browser Launcher]
            PW2[Context Factory]
            PW3[Auto-Wait Handler]
            PW4[Trace Collector]
        end
    end

    subgraph "Browser Drivers"
        BD1[ChromeDriver]
        BD2[GeckoDriver]
        BD3[EdgeDriver]
        BD4[SafariDriver]
    end

    subgraph "Playwright Browsers"
        PB1[Chromium]
        PB2[Firefox]
        PB3[WebKit]
    end

    EI --> CL1
    EI --> CL2
    EI --> CL3
    EI --> CL4

    CL1 --> SM1
    CL2 --> SM1
    CL3 --> SM1
    CL4 --> SM1

    SM1 --> SM2
    SM1 --> SM3

    SM1 --> M1
    SM1 --> M2
    SM1 --> M3
    SM1 --> M4

    SM1 --> SE1
    SM1 --> PW1

    SE1 --> SE2
    SE1 --> SE3
    SE1 --> SE4

    PW1 --> PW2
    PW1 --> PW3
    PW1 --> PW4

    SE1 --> BD1
    SE1 --> BD2
    SE1 --> BD3
    SE1 --> BD4

    PW1 --> PB1
    PW1 --> PB2
    PW1 --> PB3

    style EI fill:#FF6B6B
    style SM1 fill:#98D8C8
    style M1 fill:#F7DC6F
```

## 5. Autonomous Testing Engine Flow

```mermaid
graph TD
    Start([User Provides Requirements])
    
    Start --> Parse[Parse Requirements<br/>Extract Key Information]
    Parse --> Analyze[Analyze Application<br/>Initial Screen Capture]
    
    Analyze --> GenPlan[Generate Test Plan<br/>Using LLM]
    GenPlan --> ValidPlan{Validate Plan<br/>Feasibility Check}
    
    ValidPlan -->|Invalid| RefineReq[Request Clarification<br/>From User]
    RefineReq --> Parse
    
    ValidPlan -->|Valid| UserApprove{User Approval<br/>Required?}
    UserApprove -->|Yes| ShowPlan[Display Plan to User]
    ShowPlan --> WaitApprove{User Approves?}
    WaitApprove -->|No| RefineReq
    WaitApprove -->|Yes| Execute
    UserApprove -->|No| Execute
    
    Execute[Execute Test Plan]
    Execute --> StepLoop{More Steps?}
    
    StepLoop -->|Yes| CaptureScreen[Capture Current Screen]
    CaptureScreen --> VisionAnalysis[Vision Analysis<br/>Find Target Element]
    VisionAnalysis --> ElementFound{Element Found?}
    
    ElementFound -->|No| Retry{Retry Count < Max?}
    Retry -->|Yes| AlternateStrategy[Try Alternate Strategy]
    AlternateStrategy --> VisionAnalysis
    Retry -->|No| LogFailure[Log Step Failure]
    LogFailure --> StepLoop
    
    ElementFound -->|Yes| PlanAction[Plan Action<br/>Coordinates + Method]
    PlanAction --> ExecuteAction[Execute Action<br/>via Execution Engine]
    ExecuteAction --> VerifyAction[Verify Action Result<br/>Screenshot + Vision]
    VerifyAction --> ActionSuccess{Action Successful?}
    
    ActionSuccess -->|No| LogFailure
    ActionSuccess -->|Yes| LogSuccess[Log Step Success<br/>Save Artifacts]
    LogSuccess --> StepLoop
    
    StepLoop -->|No| GenReport[Generate Comprehensive Report]
    GenReport --> SaveArtifacts[Save All Artifacts<br/>Screenshots, Videos, Logs]
    SaveArtifacts --> End([Return Report to User])

    style Start fill:#50C878
    style Execute fill:#4A90E2
    style GenReport fill:#FFB347
    style End fill:#50C878
```

## 6. Data Flow Architecture

```mermaid
graph LR
    subgraph "Input Sources"
        I1[User Requirements]
        I2[Test Scripts YAML]
        I3[Configuration Files]
        I4[Environment Variables]
    end

    subgraph "Processing Layer"
        P1[Requirement Parser]
        P2[Test Plan Generator]
        P3[Script Interpreter]
        P4[Vision Processor]
        P5[Action Executor]
    end

    subgraph "Data Stores"
        D1[(Test Plans DB)]
        D2[(Execution History)]
        D3[(Element Cache)]
        D4[(Artifact Storage)]
    end

    subgraph "Output Artifacts"
        O1[Test Reports HTML]
        O2[Screenshots PNG]
        O3[Videos MP4]
        O4[Logs JSON]
        O5[Metrics CSV]
    end

    I1 --> P1
    I2 --> P3
    I3 --> P1
    I4 --> P5

    P1 --> P2
    P2 --> D1
    P3 --> P5

    P5 --> P4
    P4 --> D3
    P4 --> P5

    P5 --> D2
    P5 --> D4

    D1 --> O1
    D2 --> O1
    D2 --> O5
    D4 --> O2
    D4 --> O3
    D4 --> O4

    style P2 fill:#50C878
    style P4 fill:#FFB347
    style P5 fill:#FF6B6B
```

## 7. Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DE1[Local IDE]
        DE2[MCP Host]
        DE3[Test Driver MCP Server<br/>Local Instance]
        DE4[Local Browser]
        DE5[Local VLM Optional]
    end

    subgraph "CI/CD Environment"
        CI1[GitHub Actions / GitLab CI]
        CI2[Test Driver Container<br/>Podman]
        CI3[Headless Browsers]
        CI4[Cloud VLM API]
    end

    subgraph "Production Testing Environment"
        PR1[Test Orchestrator]
        PR2[Test Driver Cluster<br/>Multiple Instances]
        PR3[Browser Grid<br/>Selenium Grid / Playwright Grid]
        PR4[Shared Vision Service]
        PR5[Artifact Storage S3]
        PR6[Report Dashboard]
    end

    DE1 --> DE2
    DE2 --> DE3
    DE3 --> DE4
    DE3 --> DE5

    CI1 --> CI2
    CI2 --> CI3
    CI2 --> CI4

    PR1 --> PR2
    PR2 --> PR3
    PR2 --> PR4
    PR2 --> PR5
    PR5 --> PR6

    style DE3 fill:#4A90E2
    style CI2 fill:#50C878
    style PR2 fill:#FFB347
```

## 8. Security and Authentication Flow

```mermaid
sequenceDiagram
    participant User
    participant MCPHost
    participant MCPServer
    participant VisionAdapter
    participant CloudAPI
    participant LocalVLM

    Note over User,LocalVLM: Configuration Phase
    User->>MCPHost: Configure vision adapter preference
    MCPHost->>MCPServer: Set adapter config
    
    alt Cloud Adapter Selected
        User->>MCPHost: Provide API key (stored locally)
        MCPHost->>MCPServer: Register API key (encrypted)
        MCPServer->>VisionAdapter: Initialize cloud adapter
    else Local Adapter Selected
        User->>MCPHost: Specify model path
        MCPHost->>MCPServer: Set local model config
        MCPServer->>LocalVLM: Load model
    end

    Note over User,LocalVLM: Execution Phase
    User->>MCPHost: Start test
    MCPHost->>MCPServer: Execute with consent
    
    alt Cloud Vision
        MCPServer->>VisionAdapter: Analyze screenshot
        VisionAdapter->>CloudAPI: API call with key
        CloudAPI-->>VisionAdapter: Vision result
        VisionAdapter-->>MCPServer: Element coordinates
    else Local Vision
        MCPServer->>LocalVLM: Analyze screenshot
        LocalVLM-->>MCPServer: Element coordinates
    end

    MCPServer-->>MCPHost: Test results
    MCPHost-->>User: Display results

    Note over User,LocalVLM: No backend API key required<br/>All credentials user-managed
```

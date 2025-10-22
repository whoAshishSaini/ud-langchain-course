from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser

from schemas import AgentResponse

message = AIMessage(content="""{  "answer": "Here are three job postings for MDM using AI in India:\n\n1. **MDM IDQ Developer** at Weekday AI (YC W21) in Chennai, Tamil Nadu. Salary: ₹2,000,000 - ₹... (Posted 5 days ago)\n2. **AI Engineer** in Remote locations (listed under Master Data Management jobs on Naukri.com)\n3. **Informatica MDM Sr Data Engineer - Asst Manager** at Weekday AI (mentioned in LinkedIn Informatica MDM Support Manager jobs)",
"sources": [
    {"url": "https://in.linkedin.com/jobs/analyst-master-data-management-mdm-jobs"},
    {"url": "https://www.naukri.com/master-data-management-jobs-in-remote-29"},
    {"url": "https://in.linkedin.com/jobs/informatica-mdm-support-manager-jobs"}
  ]
}""")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
result = output_parser.invoke(message)
print(result)  # {'foo': 'bar'}

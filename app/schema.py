#Skleton of Api
from pydantic import BaseModel

class EmployeeStatus(BaseModel):
    PerfScoreID:int
    Salary:int
    PositionID:int
    EngagementSurvey:int
    EmpSatisfaction:float
    SpecialProjectsCount:int 
    DaysLateLast30:int
    Absences:int
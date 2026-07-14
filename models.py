"""
models.py
---------
All Pydantic schemas used to validate structured resume data extracted
by the LLM. Keeping every schema in one file makes it easy to see the
full "shape" of the data the app expects, and easy to reuse the same
models across app.py, extractor.py, and parser.py.

Beginner note: Pydantic models are just Python classes that describe
what fields an object should have and what type each field is. When we
feed JSON into one of these models, Pydantic checks (validates) that
the JSON matches the expected shape and gives us a nice Python object
back (or raises a clear error if something is wrong).
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# PersonalInfo schema
# ---------------------------------------------------------------------
class PersonalInfo(BaseModel):
    """Basic contact / identity information about the candidate."""
    full_name: Optional[str] = Field(default=None, description="Candidate's full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone_number: Optional[str] = Field(default=None, description="Phone number")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    github: Optional[str] = Field(default=None, description="GitHub profile URL")
    portfolio: Optional[str] = Field(default=None, description="Personal portfolio/website URL")
    address: Optional[str] = Field(default=None, description="Street address")
    city: Optional[str] = Field(default=None, description="City")
    country: Optional[str] = Field(default=None, description="Country")


# ---------------------------------------------------------------------
# ProfessionalInfo schema
# ---------------------------------------------------------------------
class ProfessionalInfo(BaseModel):
    """High-level career information."""
    current_job_title: Optional[str] = None
    years_of_experience: Optional[str] = Field(
        default=None,
        description="Total years of experience, e.g. '5 years' or '5'",
    )
    industry: Optional[str] = None
    current_company: Optional[str] = None


# ---------------------------------------------------------------------
# Skills schema
# ---------------------------------------------------------------------
class Skills(BaseModel):
    """Skills broken out by category. Each is a list of strings."""
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    libraries: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    cloud_platforms: List[str] = Field(default_factory=list)
    devops_tools: List[str] = Field(default_factory=list)
    ai_tools: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Education schema
# ---------------------------------------------------------------------
class Education(BaseModel):
    """One education record (degree)."""
    degree: Optional[str] = None
    field: Optional[str] = None
    university: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    gpa: Optional[str] = None


# ---------------------------------------------------------------------
# Experience schema
# ---------------------------------------------------------------------
class Experience(BaseModel):
    """One work-experience entry."""
    company: Optional[str] = None
    position: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    technologies_used: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Project schema
# ---------------------------------------------------------------------
class Project(BaseModel):
    """One project entry."""
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    github_link: Optional[str] = None
    live_demo: Optional[str] = None


# ---------------------------------------------------------------------
# Top-level ResumeData schema
# ---------------------------------------------------------------------
class ResumeData(BaseModel):
    """
    Top-level schema representing the entire structured resume.
    This is the object we ask the LLM to fill in, and the object
    we validate the LLM's JSON output against.
    """
    personal_information: PersonalInfo = Field(default_factory=PersonalInfo)
    professional_information: ProfessionalInfo = Field(default_factory=ProfessionalInfo)
    professional_summary: Optional[str] = None

    skills: Skills = Field(default_factory=Skills)

    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)

    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    internships: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    volunteer_experience: List[str] = Field(default_factory=list)

    class Config:
        # Allows population by field name (default anyway) and keeps
        # the model strict about extra keys not defined above.
        # -------------------------------------------------------------
        # Config: ignore any unexpected extra keys from LLM output
        # -------------------------------------------------------------
        extra = "ignore"

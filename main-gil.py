from crewai import Crew, Process

from agents import YoutubeAutomationAgents
from tasks import YoutubeAutomationTasks
from langchain_openai import ChatOpenAI

from tools.youtube_video_details_tool import YoutubeVideoDetailsTool
from tools.youtube_video_search_tool import YoutubeVideoSearchTool

from youtube_transcript_api import YouTubeTranscriptApi

from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI GPT-4 language model
OpenAIGPT = ChatOpenAI(
    model="gpt-3.5-turbo" #"gpt-4"
)

agents = YoutubeAutomationAgents()
tasks = YoutubeAutomationTasks()

youtube_video_search_tool = YoutubeVideoSearchTool()
youtube_video_details_tool = YoutubeVideoDetailsTool()

#Tools due diligence
results = youtube_video_search_tool.run(keyword="council", max_results=15)

# https://www.youtube.com/watch?v=TMBmMb4ZwFc
details = youtube_video_details_tool.run(video_id="TMBmMb4ZwFc")
print(details)

transcript = YouTubeTranscriptApi.get_transcript(video_id="TMBmMb4ZwFc")
print(transcript)

# for result in results:
#     details = youtube_video_details_tool.run(video_id=result.video_id)
#     print(details)
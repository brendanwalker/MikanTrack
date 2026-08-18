#include "MainMenuScreen.h"

#include "imgui.h"
#include "tinyfiledialogs.h"

#include "PathUtils.h"
#include "ProjectManager.h"

MainMenuScreen::Action MainMenuScreen::draw(bool bHasLastProject, const std::string& lastProjectName)
{
	Action action;

	const ImGuiViewport* viewport= ImGui::GetMainViewport();
	const ImVec2 center(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
						viewport->WorkPos.y + viewport->WorkSize.y * 0.5f);

	ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
	ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_Always);
	const ImGuiWindowFlags flags= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
								  ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
								  ImGuiWindowFlags_AlwaysAutoResize;
	ImGui::Begin("MainMenu", nullptr, flags);

	{
		const char* title= "MikanTrack";
		const float textWidth= ImGui::CalcTextSize(title).x;
		ImGui::SetCursorPosX((ImGui::GetWindowSize().x - textWidth) * 0.5f);
		ImGui::TextUnformatted(title);
	}
	ImGui::Separator();
	ImGui::Spacing();

	const ImVec2 buttonSize(-1.f, 40.f);

	if (bHasLastProject)
	{
		const std::string resumeLabel= "Resume " + lastProjectName;
		if (ImGui::Button(resumeLabel.c_str(), buttonSize))
			action.type= Action::Type::Resume;
	}

	if (ImGui::Button("New Project...", buttonSize))
	{
		m_nameBuffer[0]= '\0';
		m_bNameModalRequested= true;
	}

	if (ImGui::Button("Load Project...", buttonSize))
	{
		// Blocks the frame loop while open; fine here, nothing is streaming in
		// the menu state
		const std::string defaultDir= (ProjectManager::getProjectsRootDirectory() / "").string();
		const char* filterPatterns[]= {"project.json"};
		const char* selectedPath= tinyfd_openFileDialog(
			"Load Project", defaultDir.c_str(), 1, filterPatterns,
			"MikanTrack project (project.json)", 0);
		if (selectedPath != nullptr)
		{
			action.type= Action::Type::LoadProject;
			// tinyfd returns UTF-8 on Windows by default
			action.projectFile= PathUtils::utf8ToPath(selectedPath);
		}
	}

	if (ImGui::Button("Exit", buttonSize))
		action.type= Action::Type::Exit;

	if (!m_statusMessage.empty())
	{
		ImGui::Spacing();
		ImGui::PushTextWrapPos(0.f);
		ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", m_statusMessage.c_str());
		ImGui::PopTextWrapPos();
	}

	// New Project name modal
	if (m_bNameModalRequested)
	{
		ImGui::OpenPopup("New Project");
		m_bNameModalRequested= false;
	}
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
	if (ImGui::BeginPopupModal("New Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		ImGui::TextUnformatted("Project name:");
		if (ImGui::IsWindowAppearing())
			ImGui::SetKeyboardFocusHere();
		ImGui::SetNextItemWidth(280.f);
		const bool bEnterPressed= ImGui::InputText("##projectName", m_nameBuffer, sizeof(m_nameBuffer),
												   ImGuiInputTextFlags_EnterReturnsTrue);

		const std::string name(m_nameBuffer);
		std::string nameError;
		bool bValid= ProjectManager::isValidProjectName(name, nameError);
		if (bValid && std::filesystem::exists(ProjectManager::getProjectsRootDirectory() / name))
		{
			bValid= false;
			nameError= "A project with this name already exists";
		}
		if (!name.empty() && !bValid)
			ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", nameError.c_str());

		ImGui::BeginDisabled(!bValid);
		const bool bCreatePressed= ImGui::Button("Create", ImVec2(120, 0));
		ImGui::EndDisabled();
		ImGui::SameLine();
		const bool bCancelPressed= ImGui::Button("Cancel", ImVec2(120, 0));

		if ((bCreatePressed || bEnterPressed) && bValid)
		{
			action.type= Action::Type::NewProject;
			action.projectName= name;
			ImGui::CloseCurrentPopup();
		}
		else if (bCancelPressed)
		{
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	ImGui::End();
	return action;
}

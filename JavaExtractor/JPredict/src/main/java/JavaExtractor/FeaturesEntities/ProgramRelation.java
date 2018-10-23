package JavaExtractor.FeaturesEntities;

import java.util.ArrayList;
import java.util.function.Function;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class ProgramRelation {
	private Property m_Source;
	private Property m_Target;
	private String m_HashedPath;
	private String m_Path;
	@SuppressWarnings("FieldCanBeLocal")
	@JsonPropertyDescription
	private ArrayList<String> result;
	public static Function<String, String> s_Hasher = (s) -> Integer.toString(s.hashCode());

	public ProgramRelation(Property sourceName, Property targetName, String path) {
		m_Source = sourceName;
		m_Target = targetName;
		m_Path = path;
		m_HashedPath = s_Hasher.apply(path);
	}

	public static void setNoHash() {
		s_Hasher = (s) -> s;
	}

	public String toString() {
		return String.format("%s,%s,%s", m_Source.getName(), m_HashedPath,
				m_Target.getName());
	}

	@JsonIgnore
	public String getPath() {
		return m_Path;
	}

	@JsonIgnore
	public Property getSource() {
		return m_Source;
	}

	@JsonIgnoreProperties
	public Property getTarget() {
		return m_Target;
	}

	@JsonIgnore
	public String getHashedPath() {
		return m_HashedPath;

	}
}

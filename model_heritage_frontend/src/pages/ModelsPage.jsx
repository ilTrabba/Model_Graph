import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Search, FileText, Users, Clock, Scale, Tag, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export default function ModelsPage() {
  const [models, setModels] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/models');
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      setModels(data.models || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const filteredModels = models.filter(model =>
    model.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatNumber = (num) => {
    if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num?.toString() || '0';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ok': return 'bg-green-100 text-green-800';
      case 'processing': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getLicenseColor = (license) => {
    const colors = {
      'MIT': 'bg-green-100 text-green-800',
      'Apache-2.0': 'bg-orange-100 text-orange-800',
      'GPL-3.0': 'bg-blue-100 text-blue-800',
      'BSD-3-Clause': 'bg-purple-100 text-purple-800',
      'CC-BY-NC-4.0': 'bg-pink-100 text-pink-800',
      'Proprietary': 'bg-red-100 text-red-800'
    };
    return colors[license] || 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center">
          <p className="text-red-600">Error: {error}</p>
          <Button onClick={fetchModels} className="mt-4">Retry</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Model Catalog</h1>
        
        <div className="flex items-center space-x-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <Input
              type="text"
              placeholder="Search models..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <div className="text-sm text-gray-500">
            {filteredModels.length} of {models.length} models
          </div>
        </div>
      </div>

      {filteredModels.length === 0 ? (
        <div className="text-center py-12">
          <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {searchTerm ? 'No models found' : 'No models yet'}
          </h3>
          <p className="text-gray-500 mb-4">
            {searchTerm 
              ? 'Try adjusting your search terms'
              : 'Upload your first model to get started'
            }
          </p>
          {!searchTerm && (
            <Link to="/add-model">
              <Button>Add Your First Model</Button>
            </Link>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredModels.map((model) => (
            <Link key={model.id} to={`/models/${model.id}`}>
              <Card className="hover:shadow-lg transition-shadow cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-lg truncate">{model.name}</CardTitle>
                      {model.is_foundation_model && (
                        <Sparkles className="h-4 w-4 text-purple-600" title="Foundation Model" />
                      )}
                    </div>
                    <Badge className={getStatusColor(model.status)}>
                      {model.status}
                    </Badge>
                  </div>
                  {model.description && (
                    <CardDescription className="line-clamp-2">
                      {model.description}
                    </CardDescription>
                  )}
                </CardHeader>
                
                <CardContent>
                  <div className="space-y-2 text-sm text-gray-600">
                    <div className="flex items-center justify-between">
                      <span>Parameters:</span>
                      <span className="font-medium">
                        {formatNumber(model.total_parameters)}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span>Layers:</span>
                      <span className="font-medium">{model.layer_count || 'N/A'}</span>
                    </div>
                    
                    {model.family_id && (
                      <div className="flex items-center justify-between">
                        <span>Family:</span>
                        <Badge variant="outline" className="text-xs">
                          {model.family_id}
                        </Badge>
                      </div>
                    )}
                    
                    {model.parent_id && (
                      <div className="flex items-center space-x-1 text-blue-600">
                        <Users className="h-3 w-3" />
                        <span className="text-xs">Has parent</span>
                      </div>
                    )}
                    
                    {/* New badges row */}
                    <div className="flex flex-wrap gap-1 pt-2">
                      {model.license && (
                        <Badge className={`text-xs ${getLicenseColor(model.license)}`}>
                          <Scale className="h-3 w-3 mr-1" />
                          {model.license}
                        </Badge>
                      )}
                      {model.task && model.task.length > 0 && (
                        <Badge variant="outline" className="text-xs text-blue-700 border-blue-300">
                          <Tag className="h-3 w-3 mr-1" />
                          {model.task.length > 1 ? `${model.task.length} tasks` : model.task[0]}
                        </Badge>
                      )}
                    </div>
                    
                    <div className="flex items-center space-x-1 text-gray-400 pt-1">
                      <Clock className="h-3 w-3" />
                      <span className="text-xs">
                        {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

